"""Progress messages for dataset I/O and benchmark outputs."""

from __future__ import annotations


def log_dataset(component: str, message: str) -> None:
    """Print a tagged dataset/benchmark I/O line (``flush`` for piped/HPC logs)."""
    print(f"[dataset:{component}] {message}", flush=True)
