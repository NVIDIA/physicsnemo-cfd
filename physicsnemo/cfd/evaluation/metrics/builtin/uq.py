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

"""Pooled (reducer) uncertainty-quantification metrics.

Every metric here is a :class:`~physicsnemo.cfd.postprocessing_tools.metric_registry.ReducerMetric`:
``partial`` returns **per-case extensive sufficient statistics** (sums & counts, additive
across cases and distributed ranks) and ``finalize`` maps the globally-summed statistics to
the final per-channel value(s). This global pooling over all points — rather than a mean of
per-case values — is the statistically correct way to estimate calibration/coverage (points
per case differ).

**Space.** All statistics are computed in **physical (de-normalized) units** — the space the
wrappers denormalize ``mean`` *and* ``std`` into. NLPD carries an additive per-channel
``log(scale)`` offset vs normalized space but its cross-method ranking is unchanged.

Channels are inferred from the prediction arrays: scalar fields (e.g. ``pressure``) give one
channel; 3-vector fields (e.g. ``shear_stress``, ``velocity``) give ``_x`` / ``_y`` / ``_z``
channels. A method with no total ``std`` (deterministic wrapper) or no ``epistemic_std``
contributes nothing, so the corresponding metric finalizes to ``NaN``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterator

import numpy as np

from physicsnemo.cfd.evaluation.datasets.schema import as_distribution
from physicsnemo.cfd.postprocessing_tools.metric_registry import register_metric

_LOG_2PI = math.log(2.0 * math.pi)
#: Numerical floor on variance for the log / division terms.
_VAR_FLOOR = 1.0e-12

#: Component suffixes for 3-vector fields.
_VEC3_SUFFIXES = ("x", "y", "z")


def _to_numpy_f64(x: Any) -> np.ndarray | None:
    """Convert a NumPy array or framework tensor (e.g. ``torch.Tensor``) to ``float64`` NumPy."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    # Duck-typed torch/cupy tensor: detach + move to host without importing torch here.
    detach = getattr(x, "detach", None)
    if callable(detach):
        x = detach()
    cpu = getattr(x, "cpu", None)
    if callable(cpu):
        x = cpu()
    return np.asarray(x, dtype=np.float64)


def _iter_channels(
    gt: dict[str, Any],
    predictions: dict[str, Any],
    *,
    need_epistemic: bool,
) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
    """Yield ``(channel_name, y, mu, sigma, sigma_epi)`` per scalar channel (physical units).

    Iterates every prediction key that resolves to a :class:`FieldDistribution` with a total
    ``std`` (and, when ``need_epistemic``, an ``epistemic_std``) and has matching ground truth.
    Vector fields are split into per-component channels. Arrays are 1-D ``float64`` over points.
    """
    if not gt:
        return
    for key in predictions:
        if key not in gt or gt[key] is None:
            continue
        dist = as_distribution(predictions, key)
        if dist is None or dist.std is None:
            continue
        epi = dist.epistemic_std
        if need_epistemic and epi is None:
            continue

        y = _to_numpy_f64(gt[key])
        mu = _to_numpy_f64(dist.mean)
        sig = _to_numpy_f64(dist.std)
        epi_np = _to_numpy_f64(epi)
        if y is None or mu is None or sig is None:
            continue

        # Normalize to (N, C): scalars -> (N, 1); vectors already (N, 3).
        def _as_2d(a: np.ndarray) -> np.ndarray | None:
            if a.ndim == 1:
                return a.reshape(-1, 1)
            if a.ndim == 2:
                return a
            return None

        y2, mu2, sig2 = _as_2d(y), _as_2d(mu), _as_2d(sig)
        epi2 = _as_2d(epi_np) if epi_np is not None else None
        if y2 is None or mu2 is None or sig2 is None:
            continue
        if not (y2.shape == mu2.shape == sig2.shape):
            continue
        if epi2 is not None and epi2.shape != y2.shape:
            epi2 = None
            if need_epistemic:
                continue

        n_comp = y2.shape[1]
        for c in range(n_comp):
            if n_comp == 1:
                chan = key
            elif n_comp == 3:
                chan = f"{key}_{_VEC3_SUFFIXES[c]}"
            else:
                chan = f"{key}_{c}"
            yield (
                chan,
                y2[:, c],
                mu2[:, c],
                sig2[:, c],
                epi2[:, c] if epi2 is not None else None,
            )


class _PooledUQMetric:
    """Reducer metric over per-point Gaussian channel statistics.

    ``per_point`` returns a dict of **extensive** contributions (sums) for one channel, plus
    ``"n"`` (point count). ``finalize_channel`` maps a channel's globally-summed stats to its
    scalar value. Statistics from all channels are namespaced ``"{channel}::{stat}"`` so the
    engine can sum them across cases with no schema change.
    """

    def __init__(
        self,
        per_point: Callable[..., dict[str, float]],
        finalize_channel: Callable[[dict[str, float]], float],
        *,
        need_epistemic: bool = False,
    ) -> None:
        self._per_point = per_point
        self._finalize_channel = finalize_channel
        self._need_epistemic = need_epistemic

    def partial(self, gt: Any, predictions: Any, **_: Any) -> dict[str, float]:
        stats: dict[str, float] = {}
        for chan, y, mu, sig, epi in _iter_channels(
            gt or {}, predictions or {}, need_epistemic=self._need_epistemic
        ):
            contrib = self._per_point(y, mu, sig, epi)
            for stat, val in contrib.items():
                stats[f"{chan}::{stat}"] = stats.get(f"{chan}::{stat}", 0.0) + float(val)
        return stats

    def finalize(self, summed: dict[str, float]) -> float | dict[str, float]:
        # Regroup "{channel}::{stat}" -> {channel: {stat: value}}.
        by_channel: dict[str, dict[str, float]] = {}
        for key, val in summed.items():
            if "::" not in key:
                continue
            chan, stat = key.split("::", 1)
            by_channel.setdefault(chan, {})[stat] = val
        if not by_channel:
            return float("nan")
        out: dict[str, float] = {}
        values: list[float] = []
        for chan in sorted(by_channel):
            v = self._finalize_channel(by_channel[chan])
            out[chan] = float(v)
            if v == v:  # skip NaN in the headline mean
                values.append(float(v))
        out["mean"] = float(np.mean(values)) if values else float("nan")
        return out


# --- per-point contribution / finalize functions -------------------------------------------


def _nlpd_contrib_total(y, mu, sig, epi) -> dict[str, float]:
    err = y - mu
    var = np.clip(sig**2, _VAR_FLOOR, None)
    nlpd = 0.5 * (_LOG_2PI + np.log(var) + err**2 / var)
    return {"sum": float(nlpd.sum()), "n": float(y.size)}


def _nlpd_contrib_epistemic(y, mu, sig, epi) -> dict[str, float]:
    err = y - mu
    var = np.clip(epi**2, _VAR_FLOOR, None)
    nlpd = 0.5 * (_LOG_2PI + np.log(var) + err**2 / var)
    return {"sum": float(nlpd.sum()), "n": float(y.size)}


def _zrms_contrib(y, mu, sig, epi) -> dict[str, float]:
    err = y - mu
    var = np.clip(sig**2, _VAR_FLOOR, None)
    return {"sum_z2": float((err**2 / var).sum()), "n": float(y.size)}


def _coverage95_contrib(y, mu, sig, epi) -> dict[str, float]:
    err = y - mu
    within = (np.abs(err) <= 1.96 * sig).astype(np.float64)
    return {"within": float(within.sum()), "n": float(y.size)}


def _sharpness_total_contrib(y, mu, sig, epi) -> dict[str, float]:
    return {"sum_sig": float(sig.sum()), "n": float(y.size)}


def _sharpness_epistemic_contrib(y, mu, sig, epi) -> dict[str, float]:
    return {"sum_sig": float(epi.sum()), "n": float(y.size)}


def _mean_ratio(sum_key: str) -> Callable[[dict[str, float]], float]:
    def _f(d: dict[str, float]) -> float:
        n = d.get("n", 0.0)
        return d.get(sum_key, 0.0) / n if n > 0 else float("nan")

    return _f


def _zrms_finalize(d: dict[str, float]) -> float:
    n = d.get("n", 0.0)
    return math.sqrt(d.get("sum_z2", 0.0) / n) if n > 0 else float("nan")


# --- pointwise per-point ranking quality: Spearman(|error|, uncertainty) ---------------------
#
# "Does per-point uncertainty rank per-point error *within* a geometry?" is a rank-correlation
# question, so the right per-case diagnostic is Spearman's rho — NOT a within-case AUSE
# (sparsification only makes sense *across* geometries; see ``_SampleAUSE`` below). Returned per
# channel (+ headline ``mean``) and averaged over cases by the engine.


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Ordinal ranks (0..n-1) of ``x`` via argsort-of-argsort.

    Continuous std / error values effectively never tie, so ordinal (vs average-tie) ranks are
    fine here and avoid a SciPy dependency.
    """
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation = Pearson correlation of the ranks of ``a`` and ``b``."""
    if a.size < 2:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra**2).sum()) * float((rb**2).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


class _UncertaintyErrorSpearman:
    """Per-case Spearman(|error|, uncertainty) over :class:`FieldDistribution` predictions.

    Pointwise (per-case) metric averaged over cases by the engine. ``rank_epistemic`` correlates
    the per-point absolute error with the epistemic std (the high-contrast OOD signal); otherwise
    with the total predictive std. rho→1 means the uncertainty perfectly orders the error.
    """

    def __init__(self, *, rank_epistemic: bool) -> None:
        self._rank_epistemic = rank_epistemic

    def __call__(self, gt: Any, predictions: Any, **_: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        values: list[float] = []
        for chan, y, mu, sig, epi in _iter_channels(
            gt or {}, predictions or {}, need_epistemic=self._rank_epistemic
        ):
            unc = epi if self._rank_epistemic else sig
            if unc is None:
                continue
            rho = _spearman(np.abs(y - mu), unc)
            out[chan] = float(rho)
            if rho == rho:  # skip NaN in the headline mean
                values.append(float(rho))
        if not out:
            return {"mean": float("nan")}
        out["mean"] = float(np.mean(values)) if values else float("nan")
        return out


# --- sample-wise sparsification / AUSE (SAMPLE metric: one number per geometry) --------------
#
# AUSE answers the active-learning question at the *geometry* level: rank the dataset's
# geometries by their aggregate predicted uncertainty, drop the most-uncertain first, and check
# that the error of what remains falls (toward the oracle that drops true-worst-error first).
# This needs one (uncertainty, error) pair *per geometry* and a global sort over geometries — not
# an additive sufficient statistic — so it is a :class:`SampleMetric`: ``partial`` emits the
# per-case scalars, and ``finalize_samples`` collects them across all cases (and ranks) and
# computes AUSE once. Direct port of ``plot_field_gp_sparsification._sparsification`` / ``_ause``.


def _sparsification_curve(
    err_per_sample: np.ndarray, rank_signal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    """Full sparsification curve for one (error, uncertainty) set of per-geometry scalars.

    Returns ``(fractions, rmse_by_uncertainty, rmse_oracle, full_rmse, ause)`` where each curve
    gives the RMSE of the *retained* geometries after removing the ``fractions`` most-uncertain
    (resp. true-worst-error) ones, or ``None`` when fewer than two geometries are present.
    ``err_per_sample`` is one non-negative error scalar per geometry; ``rank_signal`` is the
    per-geometry uncertainty used to order the uncertainty-driven removals. AUSE is the
    normalized area between the uncertainty and oracle curves (lower is better; 0 == oracle).
    Direct port of ``plot_field_gp_sparsification._sparsification`` / ``_ause``.
    """
    err2 = np.asarray(err_per_sample, dtype=np.float64) ** 2
    n = err2.size
    if n < 2:
        return None
    fr = np.arange(n, dtype=np.float64) / n

    def _suffix_rmse(order: np.ndarray) -> np.ndarray:
        vals = err2[order]  # ordered so the first entries are removed first
        suffix_sum = np.cumsum(vals[::-1])[::-1]  # sum over the retained tail
        counts = n - np.arange(n)
        return np.sqrt(suffix_sum / counts)

    by_unc = _suffix_rmse(np.argsort(-np.asarray(rank_signal, dtype=np.float64)))
    oracle = _suffix_rmse(np.argsort(-err2))
    full = float(np.sqrt(err2.mean()))
    ause = (
        float(np.trapz((by_unc - oracle) / full, fr)) if full > 0.0 else float("nan")
    )
    return fr, by_unc, oracle, full, ause


def _ause_curve_area(err_per_sample: np.ndarray, rank_signal: np.ndarray) -> float:
    """Scalar AUSE only (see :func:`_sparsification_curve`); ``NaN`` for <2 geometries."""
    curve = _sparsification_curve(err_per_sample, rank_signal)
    return float("nan") if curve is None else curve[4]


def _curve_payload(
    err_per_sample: np.ndarray, rank_signal: np.ndarray
) -> dict[str, Any] | None:
    """Serializable curve dict for the sparsification visual, or ``None`` for <2 geometries."""
    curve = _sparsification_curve(err_per_sample, rank_signal)
    if curve is None:
        return None
    fr, by_unc, oracle, full, ause = curve
    return {
        "fractions": fr,
        "by_uncertainty": by_unc,
        "oracle": oracle,
        "full": full,
        "ause": ause,
        "n": int(fr.size),
    }


class _SampleAUSE:
    """Sample-wise AUSE. ``partial`` → per-geometry (uncertainty, error); ``finalize_samples`` → AUSE.

    Per channel, ``partial`` reduces one geometry to its RMS error (``err``) and its mean
    predicted std (``unc``; epistemic when ``rank_epistemic``). ``finalize_samples`` gathers those
    across all geometries and computes the AUSE of ranking geometries by ``unc``.
    """

    def __init__(self, *, rank_epistemic: bool) -> None:
        self._rank_epistemic = rank_epistemic

    def partial(self, gt: Any, predictions: Any, **_: Any) -> dict[str, float]:
        stats: dict[str, float] = {}
        for chan, y, mu, sig, epi in _iter_channels(
            gt or {}, predictions or {}, need_epistemic=self._rank_epistemic
        ):
            unc = epi if self._rank_epistemic else sig
            if unc is None:
                continue
            err = float(np.sqrt(np.mean((y - mu) ** 2)))  # per-geometry RMS error
            stats[f"{chan}::err"] = err
            stats[f"{chan}::unc"] = float(np.mean(unc))  # per-geometry mean uncertainty
        return stats

    @staticmethod
    def _regroup(collected: dict[str, list[float]]) -> dict[str, dict[str, list[float]]]:
        """Regroup ``{channel}::{err|unc}`` -> ``{channel: {"err": [...], "unc": [...]}}``."""
        by_channel: dict[str, dict[str, list[float]]] = {}
        for key, vals in collected.items():
            if "::" not in key:
                continue
            chan, stat = key.rsplit("::", 1)
            by_channel.setdefault(chan, {})[stat] = vals
        return by_channel

    def finalize_samples(
        self, collected: dict[str, list[float]]
    ) -> float | dict[str, float]:
        by_channel = self._regroup(collected)
        if not by_channel:
            return float("nan")
        out: dict[str, float] = {}
        values: list[float] = []
        for chan in sorted(by_channel):
            err = np.asarray(by_channel[chan].get("err", []), dtype=np.float64)
            unc = np.asarray(by_channel[chan].get("unc", []), dtype=np.float64)
            if err.size != unc.size or err.size < 2:
                out[chan] = float("nan")
                continue
            v = _ause_curve_area(err, unc)
            out[chan] = float(v)
            if v == v:  # skip NaN in the headline mean
                values.append(float(v))
        out["mean"] = float(np.mean(values)) if values else float("nan")
        return out

    def curves(self, collected: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
        """Per-channel sparsification curves for the plot (one series per field channel)."""
        by_channel = self._regroup(collected)
        out: dict[str, dict[str, Any]] = {}
        for chan in sorted(by_channel):
            err = np.asarray(by_channel[chan].get("err", []), dtype=np.float64)
            unc = np.asarray(by_channel[chan].get("unc", []), dtype=np.float64)
            if err.size != unc.size:
                continue
            payload = _curve_payload(err, unc)
            if payload is not None:
                out[chan] = payload
        return out


# --- sample-wise drag sparsification (SAMPLE metric): linear UQ propagation into Cd ----------
#
# Drag is a *linear* functional of the surface fields, so the predicted-drag mean is the surface
# integral of the predicted mean field and the drag *variance* is the area/normal-weighted sum of
# the per-point field variances (diagonal-posterior linear error propagation; a lower bound under
# spatial correlation, but exact-enough for *ranking* geometries — which is all sparsification
# needs). This is the decision-relevant panel: does the field UQ flag the geometries whose drag we
# predict worst? Direct port of ``field_gp_utils.compute_drag_uq_stats``, but reading physical-unit
# fields straight off the benchmark comparison mesh (so no normalization factors are needed).


def _drag_mean_and_std(
    normals: np.ndarray,
    area: np.ndarray,
    direction: np.ndarray,
    p: np.ndarray,
    wss: np.ndarray,
    p_std: np.ndarray,
    wss_std: np.ndarray,
    coeff: float,
) -> tuple[float, float]:
    """Integrated drag coefficient and its linear-propagated std for one geometry (physical units).

    ``normals`` (N,3) cell unit normals, ``area`` (N,) cell areas, ``direction`` (3,) force
    direction, ``p`` (N,) pressure, ``wss`` (N,3) wall shear stress, ``p_std`` (N,) / ``wss_std``
    (N,3) the per-cell field std to propagate. Mirrors ``compute_force_coefficients`` for the mean
    and the per-channel drag weights ``w_p = coeff*(n·f)*a`` / ``w_tau = -coeff*a*f`` for the
    variance ``Var[Cd] = Σ w_p² p_std² + Σ w_tau² wss_std²``.
    """
    n_dot_f = normals @ direction  # (N,)
    c_p = coeff * float(np.sum(n_dot_f * area * p))
    c_f = -coeff * float(np.sum((wss @ direction) * area))
    drag = c_p + c_f
    w_p = coeff * n_dot_f * area  # (N,)
    w_tau = -coeff * area[:, None] * direction[None, :]  # (N, 3)
    var = float(np.sum(w_p**2 * p_std**2)) + float(np.sum(w_tau**2 * wss_std**2))
    return drag, math.sqrt(max(var, 0.0))


def _cell_array(mesh: Any, name: str | None) -> np.ndarray | None:
    """Return ``mesh.cell_data[name]`` as ``float64`` (or ``None`` when absent)."""
    if name is None:
        return None
    try:
        cd = mesh.cell_data
        if name in cd:
            return np.asarray(cd[name], dtype=np.float64)
    except (AttributeError, KeyError, TypeError):
        return None
    return None


class _SampleDragUQ:
    """Sample-wise drag sparsification via linear UQ propagation into the ``Cd`` integral (surface).

    ``partial`` reduces one geometry to ``(drag_abs_err, drag_epistemic_std, drag_total_std)`` by
    integrating the comparison mesh's predicted / GT pressure + WSS and propagating the per-cell
    predictive std into the drag integral. ``finalize_samples`` computes the AUSE of ranking
    geometries by drag epistemic-std and by drag total-std against ``|Cd_pred - Cd_true|``.
    Requires the comparison mesh to carry cell-dof std companions (the benchmark attaches them when
    ``output.std_mesh_field_names`` / ``epistemic_std_mesh_field_names`` are set), so it is a no-op
    for deterministic wrappers or when std fields are unavailable.

    ``coeff`` (Cd prefactor ``2/(A·ρ·U²)``; only sets the overall scale, which cancels in AUSE) and
    ``drag_direction`` are configurable exactly like the deterministic ``drag`` metric
    (:func:`~physicsnemo.cfd.evaluation.metrics.builtin.forces.drag_error`): the constructor sets
    the registration-time defaults and ``partial`` accepts per-run overrides from the metric's
    config ``kwargs`` (e.g. ``- {name: drag_uq, coeff: 2.5, drag_direction: [1, 0, 0]}``).
    """

    def __init__(
        self,
        *,
        coeff: float = 1.0,
        drag_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> None:
        self._coeff = float(coeff)
        self._dir = tuple(float(x) for x in drag_direction)

    def partial(
        self,
        gt: Any,
        predictions: Any,
        *,
        case: Any = None,
        comparison_mesh: Any = None,
        metric_dtype: str | None = None,
        output: Any = None,
        coeff: float | None = None,
        drag_direction: list[float] | None = None,
        **_: Any,
    ) -> dict[str, float]:
        predictions = predictions or {}
        # Per-run overrides from config kwargs, else the registration-time defaults.
        used_coeff = self._coeff if coeff is None else float(coeff)
        direction = np.asarray(
            self._dir if drag_direction is None else drag_direction, dtype=np.float64
        )
        pdist = as_distribution(predictions, "pressure")
        wdist = as_distribution(predictions, "shear_stress")
        # Deterministic wrapper (no std) or missing fields: nothing to propagate.
        if (
            output is None
            or pdist is None
            or wdist is None
            or pdist.std is None
            or wdist.std is None
        ):
            return {}
        from physicsnemo.cfd.evaluation.metrics.mesh_bridge import (
            resolve_comparison_mesh_for_metric,
        )

        mesh, dtype = resolve_comparison_mesh_for_metric(
            predictions,
            case=case,
            comparison_mesh=comparison_mesh,
            metric_dtype=metric_dtype,
            output=output,
        )
        # Drag integration here uses explicit *cell* normals/areas; require the cell dof (the
        # config's surface_interpolate_point_to_cell_for_metrics guarantees it).
        if mesh is None or dtype != "cell":
            return {}

        prp = output.mesh_field_names.get("pressure")
        prw = output.mesh_field_names.get("shear_stress")
        gtp = output.ground_truth_mesh_field_names.get("pressure")
        gtw = output.ground_truth_mesh_field_names.get("shear_stress")
        std_p = output.std_mesh_field_names.get("pressure")
        std_w = output.std_mesh_field_names.get("shear_stress")
        epi_p = output.epistemic_std_mesh_field_names.get("pressure")
        epi_w = output.epistemic_std_mesh_field_names.get("shear_stress")

        # Explicit per-cell normals + areas (physically correct surface integral; consistent
        # weights for mean and variance).
        geom = mesh.compute_normals(
            cell_normals=True, point_normals=False, inplace=False
        )
        geom = geom.compute_cell_sizes(length=False, area=True, volume=False)
        normals = np.asarray(geom.cell_data["Normals"], dtype=np.float64)
        area = np.asarray(geom.cell_data["Area"], dtype=np.float64).reshape(-1)

        p_pred = _cell_array(geom, prp)
        wss_pred = _cell_array(geom, prw)
        p_true = _cell_array(geom, gtp)
        wss_true = _cell_array(geom, gtw)
        p_std = _cell_array(geom, std_p)
        wss_std = _cell_array(geom, std_w)
        p_epi = _cell_array(geom, epi_p)
        wss_epi = _cell_array(geom, epi_w)
        if any(
            a is None
            for a in (p_pred, wss_pred, p_true, wss_true, p_std, wss_std)
        ):
            return {}

        drag_pred, drag_total_std = _drag_mean_and_std(
            normals, area, direction, p_pred, wss_pred, p_std, wss_std, used_coeff
        )
        drag_true, _ = _drag_mean_and_std(
            normals, area, direction, p_true, wss_true,
            np.zeros_like(p_true), np.zeros_like(wss_true), used_coeff,
        )
        stats = {
            "abs_err": abs(drag_pred - drag_true),
            "total_std": drag_total_std,
        }
        if p_epi is not None and wss_epi is not None:
            _, drag_epi_std = _drag_mean_and_std(
                normals, area, direction, p_pred, wss_pred, p_epi, wss_epi, used_coeff
            )
            stats["epi_std"] = drag_epi_std
        return stats

    def finalize_samples(
        self, collected: dict[str, list[float]]
    ) -> float | dict[str, float]:
        err = np.asarray(collected.get("abs_err", []), dtype=np.float64)
        out: dict[str, float] = {}
        for sub, key in (("epistemic", "epi_std"), ("total", "total_std")):
            unc = np.asarray(collected.get(key, []), dtype=np.float64)
            if err.size == unc.size and err.size >= 2:
                out[sub] = _ause_curve_area(err, unc)
            else:
                out[sub] = float("nan")
        return out

    def curves(self, collected: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
        """Drag sparsification curves ranked by epistemic and total drag std."""
        err = np.asarray(collected.get("abs_err", []), dtype=np.float64)
        out: dict[str, dict[str, Any]] = {}
        for sub, key in (("epistemic", "epi_std"), ("total", "total_std")):
            unc = np.asarray(collected.get(key, []), dtype=np.float64)
            if err.size != unc.size:
                continue
            payload = _curve_payload(err, unc)
            if payload is not None:
                out[sub] = payload
        return out


def _make_uq_metrics() -> dict[str, _PooledUQMetric]:
    """Instantiate the pooled UQ reducer metrics (shared across surface/volume domains)."""
    return {
        "nlpd": _PooledUQMetric(_nlpd_contrib_total, _mean_ratio("sum")),
        "nlpd_epistemic": _PooledUQMetric(
            _nlpd_contrib_epistemic, _mean_ratio("sum"), need_epistemic=True
        ),
        "calibration_zrms": _PooledUQMetric(_zrms_contrib, _zrms_finalize),
        "coverage_95": _PooledUQMetric(_coverage95_contrib, _mean_ratio("within")),
        "sharpness_std": _PooledUQMetric(
            _sharpness_total_contrib, _mean_ratio("sum_sig")
        ),
        "sharpness_epistemic_std": _PooledUQMetric(
            _sharpness_epistemic_contrib, _mean_ratio("sum_sig"), need_epistemic=True
        ),
    }


def _make_pointwise_uq_metrics() -> dict[str, _UncertaintyErrorSpearman]:
    """Instantiate the pointwise (per-case) UQ metrics: Spearman(|error|, uncertainty)."""
    return {
        "uncertainty_error_spearman": _UncertaintyErrorSpearman(rank_epistemic=False),
        "uncertainty_error_spearman_epistemic": _UncertaintyErrorSpearman(
            rank_epistemic=True
        ),
    }


def _make_sample_uq_metrics() -> dict[str, Any]:
    """Instantiate the sample-wise (per-geometry) UQ metrics: field + drag sparsification AUSE."""
    return {
        "sparsification_ause": _SampleAUSE(rank_epistemic=False),
        "sparsification_ause_epistemic": _SampleAUSE(rank_epistemic=True),
    }


def register_uq_metrics() -> None:
    """Register the pooled (reducer), pointwise, and sample-wise UQ metrics for both domains."""
    for domain in ("surface", "volume"):
        for name, metric in _make_uq_metrics().items():
            register_metric(name, metric, domain=domain)
        for name, metric in _make_pointwise_uq_metrics().items():
            register_metric(name, metric, domain=domain)
        for name, metric in _make_sample_uq_metrics().items():
            register_metric(name, metric, domain=domain)
    # Drag UQ integrates a surface force, so it is registered for the surface domain only.
    register_metric("drag_uq", _SampleDragUQ(), domain="surface")
