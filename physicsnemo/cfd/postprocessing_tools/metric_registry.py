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

"""Named metric registry with optional domain scoping (surface / volume).

Two metric kinds share this one registry:

- **Pointwise** metrics are plain callables ``fn(gt, predictions, **extended) -> float |
  dict[str, float]``. The engine collects one value per case and aggregates by mean over
  cases. All deterministic (L2 / force / physics) metrics are this kind.
- **Pooled / reducer** metrics implement the :class:`ReducerMetric` protocol
  (``partial`` + ``finalize``). The engine calls ``partial`` per case, **sums** the
  returned extensive sufficient statistics across all cases (and across distributed
  ranks), and calls ``finalize`` once per (model, dataset). This way, the pooling of
  calibration/coverage metrics over all points is statistically correct.
- **Sample** metrics implement the :class:`SampleMetric` protocol (``partial`` +
  ``finalize_samples``). Like reducers, ``partial`` runs per case, but the engine
  **collects** (does not sum) the returned per-geometry scalars across all cases and
  ranks, and ``finalize_samples`` maps that list to the final value(s). Used for metrics
  that need a global sort over geometries — e.g. sample-wise sparsification / AUSE.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

MetricFn = Callable[..., float | dict[str, float]]


@runtime_checkable
class ReducerMetric(Protocol):
    """A pooled metric expressed as two pure functions over additive sufficient statistics."""

    def partial(self, gt: Any, predictions: Any, **extended: Any) -> dict[str, float]:
        """Per-case **extensive** sufficient statistics (sums & counts; additive across cases)."""
        ...

    def finalize(self, summed: dict[str, float]) -> float | dict[str, float]:
        """Map globally-summed statistics to the final metric value(s)."""
        ...


@runtime_checkable
class SampleMetric(Protocol):
    """A per-geometry metric whose per-case values are collected (not summed) then finalized."""

    def partial(self, gt: Any, predictions: Any, **extended: Any) -> dict[str, float]:
        """Per-case (per-geometry) scalar contributions (collected across cases, not summed)."""
        ...

    def finalize_samples(
        self, collected: dict[str, list[float]]
    ) -> float | dict[str, float]:
        """Map the per-key lists of per-geometry values to the final metric value(s)."""
        ...


#: A registry entry is a pointwise callable, a reducer-metric, or a sample-metric instance.
MetricEntry = "MetricFn | ReducerMetric | SampleMetric"

_REGISTRY: dict[tuple[str, str | None], Any] = {}


def is_reducer_metric(obj: Any) -> bool:
    """True if ``obj`` is a pooled reducer metric (has callable ``partial`` and ``finalize``).

    Checked structurally (not ``isinstance(..., ReducerMetric)``) so plain metric callables
    — which are not classes with these methods — are never misclassified.
    """
    return callable(getattr(obj, "partial", None)) and callable(
        getattr(obj, "finalize", None)
    )


def is_sample_metric(obj: Any) -> bool:
    """True if ``obj`` is a sample metric (has callable ``partial`` and ``finalize_samples``).

    Disjoint from :func:`is_reducer_metric` in practice: a sample metric exposes
    ``finalize_samples`` (not ``finalize``), so the engine's collect-not-sum path is selected.
    """
    return callable(getattr(obj, "partial", None)) and callable(
        getattr(obj, "finalize_samples", None)
    )


def register_metric(name: str, fn: Any, *, domain: str | None = None) -> None:
    """Register a metric by name and optional domain.

    ``fn`` is either a pointwise callable or a :class:`ReducerMetric` instance.

    When ``domain`` is ``None`` the metric is domain-agnostic and acts as the
    fallback when no domain-specific variant exists.  When set (e.g.
    ``"surface"`` or ``"volume"``), the metric is scoped to that domain and
    only selected when the engine resolves with a matching domain.
    """
    _REGISTRY[(name, domain)] = fn


def unregister_metric(name: str, *, domain: str | None = None) -> None:
    """Remove a metric entry (e.g. test teardown). No-op if the key is missing."""
    _REGISTRY.pop((name, domain), None)


def get_metric(name: str, *, domain: str | None = None) -> Any:
    """Resolve a metric function by name, preferring a domain-specific variant.

    Lookup order:
    1. ``(name, domain)`` — exact domain match.
    2. ``(name, None)`` — domain-agnostic fallback.
    """
    key = (name, domain)
    if key in _REGISTRY:
        return _REGISTRY[key]
    fallback = (name, None)
    if fallback in _REGISTRY:
        return _REGISTRY[fallback]
    available = sorted({n for n, _ in _REGISTRY})
    raise KeyError(
        f"Unknown metric: {name!r} (domain={domain!r}). Available: {available}"
    )


def list_metrics() -> list[str]:
    """Return sorted unique metric names (without domain qualifiers)."""
    return sorted({name for name, _ in _REGISTRY})
