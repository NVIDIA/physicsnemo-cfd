"""Uniform progress messages for inference (wrappers and CLI)."""

from __future__ import annotations


def log_inference(component: str, message: str) -> None:
    """Print a tagged inference progress line (flush for HPC / piped logs)."""
    print(f"[inference:{component}] {message}", flush=True)
