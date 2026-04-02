"""Stub adapter for Ahmed Body dataset (placeholder until implementation)."""

from pathlib import Path

from physicsnemo.cfd.evaluation.datasets.adapter_registry import DatasetAdapter
from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase


class AhmedAdapter(DatasetAdapter):
    """Placeholder for Ahmed Body dataset. Implement list_cases and load_case."""

    def __init__(self, root: str, **kwargs: object) -> None:
        self.root = Path(root)
        if self.root.exists():
            pass  # optional: validate layout

    def list_cases(self, split: str | None = None) -> list[str]:
        """Return case IDs when dataset is implemented."""
        if not self.root.exists():
            return []
        # Stub: no cases until real layout is defined
        return []

    def load_case(self, case_id: str) -> CanonicalCase:
        """Load one case. Raises until dataset layout is implemented."""
        raise NotImplementedError(
            "Ahmed Body adapter not yet implemented. "
            "Add mesh layout and GT loading in datasets/adapters/ahmed.py"
        )
