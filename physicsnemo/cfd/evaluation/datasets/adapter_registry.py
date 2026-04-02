"""Dataset adapter base class and registry."""

from abc import ABC, abstractmethod
from typing import Any, Type

from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase, InferenceDomain

_REGISTRY: dict[str, Type["DatasetAdapter"]] = {}


class DatasetAdapter(ABC):
    """Abstract interface for CFD dataset adapters."""

    @classmethod
    def inference_domain(cls) -> InferenceDomain:
        """Default domain when not configured via kwargs (subclasses may override)."""
        return "surface"

    @classmethod
    def inference_domain_from_kwargs(cls, kwargs: dict[str, Any] | None) -> InferenceDomain:
        """Domain for this adapter given ``dataset.kwargs`` (used before instantiation).

        Adapters that switch surface/volume via kwargs (e.g. DrivAerML) must override this.
        """
        return cls.inference_domain()

    @abstractmethod
    def list_cases(self, split: str | None = None) -> list[str]:
        """Return case IDs (e.g. run IDs or file stems)."""
        ...

    @abstractmethod
    def load_case(self, case_id: str) -> CanonicalCase:
        """Load mesh(es), flow conditions, and optional ground truth into canonical schema."""
        ...

    def validate(self, case: CanonicalCase) -> bool:
        """Check required fields and mesh validity. Override as needed."""
        return case.mesh_path != "" and case.mesh_type in ("point", "cell")


def register_adapter(name: str, adapter_class: Type[DatasetAdapter]) -> None:
    """Register a dataset adapter by name."""
    _REGISTRY[name] = adapter_class


def get_adapter(name: str) -> Type[DatasetAdapter]:
    """Resolve adapter class by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset adapter: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_adapters() -> list[str]:
    """Return registered adapter names."""
    return list(_REGISTRY.keys())
