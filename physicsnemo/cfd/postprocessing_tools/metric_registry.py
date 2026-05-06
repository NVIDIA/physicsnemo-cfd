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

"""Named metric registry with optional domain scoping (surface / volume)."""

from __future__ import annotations

from typing import Any, Callable

MetricFn = Callable[..., float | dict[str, float]]

_REGISTRY: dict[tuple[str, str | None], MetricFn] = {}


def register_metric(name: str, fn: MetricFn, *, domain: str | None = None) -> None:
    """Register a metric by name and optional domain.

    When ``domain`` is ``None`` the metric is domain-agnostic and acts as the
    fallback when no domain-specific variant exists.  When set (e.g.
    ``"surface"`` or ``"volume"``), the metric is scoped to that domain and
    only selected when the engine resolves with a matching domain.
    """
    _REGISTRY[(name, domain)] = fn


def unregister_metric(name: str, *, domain: str | None = None) -> None:
    """Remove a metric entry (e.g. test teardown). No-op if the key is missing."""
    _REGISTRY.pop((name, domain), None)


def get_metric(name: str, *, domain: str | None = None) -> MetricFn:
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
