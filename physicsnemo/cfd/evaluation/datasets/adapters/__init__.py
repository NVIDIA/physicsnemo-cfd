"""Dataset adapter implementations."""

from physicsnemo.cfd.evaluation.datasets.adapter_registry import register_adapter
from physicsnemo.cfd.evaluation.datasets.adapters.drivaerml import DrivAerMLAdapter
from physicsnemo.cfd.evaluation.datasets.adapters.ahmed import AhmedAdapter

register_adapter("drivaerml", DrivAerMLAdapter)
register_adapter("ahmed", AhmedAdapter)

__all__ = ["DrivAerMLAdapter", "AhmedAdapter"]
