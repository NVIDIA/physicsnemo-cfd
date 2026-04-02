"""Dataset adapters and canonical case schema."""

from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase, predictions_dict
from physicsnemo.cfd.evaluation.datasets.adapter_registry import (
    get_adapter,
    list_adapters,
    register_adapter,
    DatasetAdapter,
)
import physicsnemo.cfd.evaluation.datasets.adapters  # noqa: F401 - register drivaerml, etc.

__all__ = [
    "CanonicalCase",
    "predictions_dict",
    "DatasetAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
