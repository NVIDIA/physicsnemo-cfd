"""Metric registry and bench-backed CFD metrics."""

from physicsnemo.cfd.evaluation.metrics.registry import (
    get_metric,
    list_metrics,
    register_metric,
)

import physicsnemo.cfd.evaluation.metrics.builtin  # noqa: F401 - register built-in metrics

__all__ = ["register_metric", "get_metric", "list_metrics"]
