# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named metric registry (used by physicsnemo.cfd.evaluation and workflows)."""

from __future__ import annotations

from typing import Any, Callable

MetricFn = Callable[..., float | dict[str, float]]

_REGISTRY: dict[str, MetricFn] = {}


def register_metric(name: str, fn: MetricFn) -> None:
    """Register a metric by name."""
    _REGISTRY[name] = fn


def get_metric(name: str) -> MetricFn:
    """Resolve metric function by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric: {name}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_metrics() -> list[str]:
    """Return registered metric names."""
    return sorted(_REGISTRY.keys())
