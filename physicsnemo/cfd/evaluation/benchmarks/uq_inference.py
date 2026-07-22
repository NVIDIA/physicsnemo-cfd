# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Engine-side UQ plumbing: the sampling inference loop and pooled reducer finalization.

1. :func:`run_sampling_inference` drives ``N`` stochastic passes for a
   ``UQ_METHOD="sampling"`` wrapper (MC-Dropout, ensembles) and
   streams them into a :class:`~physicsnemo.cfd.evaluation.datasets.schema.FieldDistribution`
   per field via Welford aggregation (no ``N`` full fields held in memory). ``prepare_inputs``
   is *not* repeated — only the forward pass (and per-pass decode).

2. The pooled **reducer** metrics emit per-case additive sufficient
   statistics. The engine stores each such statistic in the per-case ``metrics`` dict under a
   reserved ``_uq::`` prefix so it flows through the existing per-case cache and distributed
   merge (both handle per-case scalars). :func:`finalize_reducer_metrics` sums those across all
   cases and calls each metric's ``finalize`` once to get the pooled value(s).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from physicsnemo.cfd.evaluation.common.inference_seed import seed_inference_rng
from physicsnemo.cfd.evaluation.datasets.schema import (
    FieldDistribution,
    build_predictive_distribution,
)

# --------------------------------------------------------------------------------------------
# Reducer partial statistics: reserved per-case keys
# --------------------------------------------------------------------------------------------

#: Per-case ``metrics`` keys carrying a reducer metric's extensive sufficient statistics are
#: namespaced with this prefix so they (a) cache like any per-case scalar, (b) survive the
#: distributed merge, and (c) are excluded from the deterministic mean-over-cases aggregation
#: and from the final per-case report rows.
REDUCER_PARTIAL_PREFIX = "_uq::"


def make_reducer_partial_key(metric_name: str, partial_key: str) -> str:
    """Reserved per-case key for one reducer statistic (``_uq::{metric}::{stat}``)."""
    return f"{REDUCER_PARTIAL_PREFIX}{metric_name}::{partial_key}"


def is_reducer_partial_key(key: str) -> bool:
    """True for keys produced by :func:`make_reducer_partial_key`."""
    return key.startswith(REDUCER_PARTIAL_PREFIX)


def _split_reducer_partial_key(key: str) -> tuple[str, str]:
    """Split ``_uq::{metric}::{stat}`` into ``(metric_name, stat_key)``."""
    body = key[len(REDUCER_PARTIAL_PREFIX) :]
    metric_name, _, partial_key = body.partition("::")
    return metric_name, partial_key


def finalize_reducer_metrics(
    per_case_rows: list[dict[str, Any]],
    domain: str | None,
    configured_metric_names: list[str] | None = None,
) -> dict[str, float]:
    """Sum reducer sufficient statistics over cases and finalize to pooled metric value(s).

    Reads the reserved ``_uq::`` keys off each row's ``metrics`` dict, sums them per
    ``(metric, stat)``, resolves each metric from the registry, and calls ``finalize``. Dict
    returns expand to ``f"{metric}_{subkey}"`` (matching the engine's pointwise expansion).
    Returns the mapping of finalized metric key -> value to merge into the run summary.

    ``configured_metric_names`` (the metrics requested in the run config) drives NaN placeholders:
    any configured reducer metric that produced **no** partials this run — e.g. a deterministic
    wrapper emits no ``_uq::`` keys — is still finalized via ``finalize({})`` so the row reports
    ``nlpd``/``coverage``/… as ``NaN`` rather than silently omitting them (consistent report
    schema; ``fail_on_any_metric_nan`` can then flag an unavailable configured metric).
    """
    # Resolve from the lightweight registry (no torch-backed metrics-package import needed;
    # the metric instances were registered there when the builtin metrics loaded).
    from physicsnemo.cfd.postprocessing_tools.metric_registry import (
        get_metric,
        is_reducer_metric,
    )

    summed: dict[str, dict[str, float]] = {}
    for row in per_case_rows:
        for key, val in (row.get("metrics") or {}).items():
            if not is_reducer_partial_key(key):
                continue
            metric_name, partial_key = _split_reducer_partial_key(key)
            if not partial_key:
                continue
            summed.setdefault(metric_name, {})
            summed[metric_name][partial_key] = (
                summed[metric_name].get(partial_key, 0.0) + float(val)
            )

    # Finalize every metric that produced partials, plus any configured reducer metric that did
    # not (with empty stats -> NaN) so deterministic rows keep a consistent schema.
    stats_by_metric: dict[str, dict[str, float]] = dict(summed)
    for name in configured_metric_names or ():
        stats_by_metric.setdefault(name, {})

    out: dict[str, float] = {}
    for metric_name, stats in stats_by_metric.items():
        try:
            metric = get_metric(metric_name, domain=domain)
        except KeyError:
            continue
        if not is_reducer_metric(metric):
            continue
        result = metric.finalize(stats)
        if isinstance(result, dict):
            for sub, v in result.items():
                out[f"{metric_name}_{sub}" if sub else metric_name] = float(v)
        else:
            out[metric_name] = float(result)
    return out


# --------------------------------------------------------------------------------------------
# Sample metric partials: reserved per-case keys (collected, not summed)
# --------------------------------------------------------------------------------------------

#: Per-case keys carrying a :class:`SampleMetric`'s per-geometry scalars. Distinct from the
#: reducer prefix so the sum-based finalize ignores them and the collect-based finalize picks
#: them up. ``"_uqs::".startswith("_uq::")`` is False, so the two namespaces are disjoint.
SAMPLE_PARTIAL_PREFIX = "_uqs::"


def make_sample_partial_key(metric_name: str, partial_key: str) -> str:
    """Reserved per-case key for one sample-metric scalar (``_uqs::{metric}::{stat}``)."""
    return f"{SAMPLE_PARTIAL_PREFIX}{metric_name}::{partial_key}"


def is_sample_partial_key(key: str) -> bool:
    """True for keys produced by :func:`make_sample_partial_key`."""
    return key.startswith(SAMPLE_PARTIAL_PREFIX)


def _split_sample_partial_key(key: str) -> tuple[str, str]:
    """Split ``_uqs::{metric}::{stat}`` into ``(metric_name, stat_key)``."""
    body = key[len(SAMPLE_PARTIAL_PREFIX) :]
    metric_name, _, partial_key = body.partition("::")
    return metric_name, partial_key


def finalize_sample_metrics(
    per_case_rows: list[dict[str, Any]],
    domain: str | None,
    configured_metric_names: list[str] | None = None,
) -> dict[str, float]:
    """Collect sample-metric per-geometry scalars over cases and finalize (e.g. sample-wise AUSE).

    Unlike :func:`finalize_reducer_metrics`, per-case values are **collected into lists** (not
    summed): each ``_uqs::`` key contributes one value per case, appended in ``per_case`` order.
    ``finalize_samples`` maps the per-key lists to the final value(s); dict returns expand to
    ``f"{metric}_{subkey}"``.

    ``configured_metric_names`` drives NaN placeholders the same way as
    :func:`finalize_reducer_metrics`: a configured sample metric with no partials this run (e.g.
    a deterministic wrapper) is finalized with an empty mapping so AUSE / drag_uq report ``NaN``
    instead of being dropped from the row.
    """
    from physicsnemo.cfd.postprocessing_tools.metric_registry import (
        get_metric,
        is_sample_metric,
    )

    collected: dict[str, dict[str, list[float]]] = {}
    for row in per_case_rows:
        for key, val in (row.get("metrics") or {}).items():
            if not is_sample_partial_key(key):
                continue
            metric_name, partial_key = _split_sample_partial_key(key)
            if not partial_key:
                continue
            collected.setdefault(metric_name, {}).setdefault(partial_key, []).append(
                float(val)
            )

    stats_by_metric: dict[str, dict[str, list[float]]] = dict(collected)
    for name in configured_metric_names or ():
        stats_by_metric.setdefault(name, {})

    out: dict[str, float] = {}
    for metric_name, stats in stats_by_metric.items():
        try:
            metric = get_metric(metric_name, domain=domain)
        except KeyError:
            continue
        if not is_sample_metric(metric):
            continue
        result = metric.finalize_samples(stats)
        if isinstance(result, dict):
            for sub, v in result.items():
                out[f"{metric_name}_{sub}" if sub else metric_name] = float(v)
        else:
            out[metric_name] = float(result)
    return out


def compute_sparsification_payload(
    per_case_rows: list[dict[str, Any]], domain: str | None
) -> dict[str, dict[str, Any]]:
    """Build the per-metric sparsification curves consumed by the ``sparsification_plot`` visual.

    Collects each sample metric's per-geometry scalars (the same reserved ``_uqs::`` keys that
    :func:`finalize_sample_metrics` uses), and for every sample metric exposing a ``curves`` method
    calls it once to produce ``{series_name: {fractions, by_uncertainty, oracle, full, ause, n}}``.
    Returns ``{metric_name: {series_name: curve_dict}}`` (empty when no sample metrics ran). Must be
    called *before* :func:`strip_reducer_partials` removes the reserved keys.
    """
    from physicsnemo.cfd.postprocessing_tools.metric_registry import (
        get_metric,
        is_sample_metric,
    )

    collected: dict[str, dict[str, list[float]]] = {}
    for row in per_case_rows:
        for key, val in (row.get("metrics") or {}).items():
            if not is_sample_partial_key(key):
                continue
            metric_name, partial_key = _split_sample_partial_key(key)
            if not partial_key:
                continue
            collected.setdefault(metric_name, {}).setdefault(partial_key, []).append(
                float(val)
            )

    out: dict[str, dict[str, Any]] = {}
    for metric_name, stats in collected.items():
        try:
            metric = get_metric(metric_name, domain=domain)
        except KeyError:
            continue
        if not is_sample_metric(metric):
            continue
        curves_fn = getattr(metric, "curves", None)
        if not callable(curves_fn):
            continue
        series = curves_fn(stats)
        if series:
            out[metric_name] = series
    return out


def is_uq_partial_key(key: str) -> bool:
    """True for any reserved UQ per-case key (reducer sufficient stat or sample-metric scalar)."""
    return is_reducer_partial_key(key) or is_sample_partial_key(key)


def select_inference_path(
    *, supports_uq: bool, uq_method: str, uq_enabled: bool
) -> str:
    """Engine per-case dispatch: ``"sampling"`` | ``"analytic"`` | ``"deterministic"``.

    ``uq_enabled`` (``run.uq.enabled``) is the master switch: when it is off, EVERY wrapper takes
    the deterministic path regardless of ``SUPPORTS_UQ`` / ``UQ_METHOD`` — so an analytic GP head
    is not executed as a distribution and produces no UQ metrics, matching the documented behavior
    and enabling apples-to-apples deterministic comparison runs.
    """
    if uq_enabled and supports_uq and uq_method == "sampling":
        return "sampling"
    if uq_enabled and supports_uq and uq_method == "analytic":
        return "analytic"
    return "deterministic"


def strip_reducer_partials(per_case_rows: list[dict[str, Any]]) -> None:
    """Remove reserved ``_uq::`` / ``_uqs::`` statistics from per-case ``metrics`` in place.

    Called once after aggregation / merge so the reported per-case rows show only real metric
    values, not the internal statistics (already folded into the run summary by
    :func:`finalize_reducer_metrics` / :func:`finalize_sample_metrics`).
    """
    for row in per_case_rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for key in [k for k in metrics if is_uq_partial_key(k)]:
            del metrics[key]


# --------------------------------------------------------------------------------------------
# Sampling inference loop (Welford streaming)
# --------------------------------------------------------------------------------------------


class _Welford:
    """Streaming mean / variance across passes for one field (arbitrary array shape).

    Tracks the across-pass (epistemic) statistics of the per-pass point predictions, plus an
    optional running sum of per-pass aleatoric variances (when a pass is itself a
    distribution), so the total predictive variance can be combined via the law of total
    variance: ``total_var = mean_i(sigma_i^2) + var_i(mu_i)``.
    """

    def __init__(self) -> None:
        self.n = 0
        self.mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._aleatoric_var_sum: np.ndarray | None = None
        self._has_aleatoric = False

    def update(self, x: np.ndarray, aleatoric_var: np.ndarray | None = None) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        if self.mean is None:
            self.mean = x.copy()
            self._m2 = np.zeros_like(x)
        else:
            delta = x - self.mean
            self.mean += delta / self.n
            self._m2 += delta * (x - self.mean)
        if aleatoric_var is not None:
            av = np.asarray(aleatoric_var, dtype=np.float64)
            self._has_aleatoric = True
            if self._aleatoric_var_sum is None:
                self._aleatoric_var_sum = av.copy()
            else:
                self._aleatoric_var_sum += av

    def result(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Return ``(mean, total_std, epistemic_std, aleatoric_std_or_None)`` (physical units)."""
        if self.n == 0 or self.mean is None:
            return None
        # Population variance across passes = epistemic uncertainty of the prediction.
        epi_var = self._m2 / self.n if self._m2 is not None else np.zeros_like(self.mean)
        epi_var = np.clip(epi_var, 0.0, None)
        epistemic_std = np.sqrt(epi_var)
        if self._has_aleatoric and self._aleatoric_var_sum is not None:
            aleatoric_var = self._aleatoric_var_sum / self.n
            aleatoric_std = np.sqrt(np.clip(aleatoric_var, 0.0, None))
            total_std = np.sqrt(np.clip(epi_var + aleatoric_var, 0.0, None))
        else:
            aleatoric_std = None
            total_std = epistemic_std
        return self.mean, total_std, epistemic_std, aleatoric_std


def _decode_pass_to_arrays(
    wrapper: Any, raw: Any, case: Any, model_input: Any
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Decode one pass to ``{field: (mean_array, aleatoric_var_or_None)}`` (physical units)."""
    preds = wrapper.decode_outputs(raw, case, model_input)
    out: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for key, value in preds.items():
        if isinstance(value, FieldDistribution):
            mean = np.asarray(value.mean, dtype=np.float64)
            ale_var = None
            if value.std is not None:
                ale_var = np.asarray(value.std, dtype=np.float64) ** 2
            out[key] = (mean, ale_var)
        else:
            out[key] = (np.asarray(value, dtype=np.float64), None)
    return out


def run_sampling_inference(
    wrapper: Any,
    case: Any,
    model_input: Any,
    *,
    n: int,
    run_seed: int,
    case_id: str,
    retain_samples: bool = False,
) -> dict[str, FieldDistribution]:
    """Drive ``n`` stochastic passes and aggregate to a distribution per field.

    Uses ``wrapper.predict_ensemble(model_input, n)`` when available (single batched call),
    else calls ``wrapper.predict(model_input)`` ``n`` times, reseeding per pass from
    ``(run_seed, case_id, pass_index)`` so each pass has a distinct, reproducible RNG state.
    """
    if n < 1:
        raise ValueError(f"num_samples must be >= 1 for sampling inference, got {n}")

    ensemble = wrapper.predict_ensemble(model_input, n)

    accumulators: dict[str, _Welford] = {}
    samples: dict[str, list[np.ndarray]] = {}

    def _consume(raw: Any) -> None:
        for key, (mean, ale_var) in _decode_pass_to_arrays(
            wrapper, raw, case, model_input
        ).items():
            accumulators.setdefault(key, _Welford()).update(mean, ale_var)
            if retain_samples:
                samples.setdefault(key, []).append(mean)

    if ensemble is not None:
        for raw in ensemble:
            _consume(raw)
    else:
        for i in range(n):
            seed_inference_rng(run_seed, f"{case_id}#pass{i}")
            _consume(wrapper.predict(model_input))

    distributions: dict[str, FieldDistribution] = {}
    for key, acc in accumulators.items():
        res = acc.result()
        if res is None:
            continue
        mean, total_std, epistemic_std, aleatoric_std = res
        stacked = (
            np.stack(samples[key], axis=0)
            if retain_samples and key in samples
            else None
        )
        distributions[key] = build_predictive_distribution(
            mean=mean,
            std=total_std,
            epistemic_std=epistemic_std,
            aleatoric_std=aleatoric_std,
            samples=stacked,
        )
    return distributions
