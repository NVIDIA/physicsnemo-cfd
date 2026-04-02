"""PyTorch 2.6+ compatibility for loading full training checkpoints (not weights-only).

``physicsnemo.utils.load_checkpoint`` uses ``torch.load`` without ``weights_only``.
Since PyTorch 2.6 the default is ``weights_only=True``, which fails on checkpoints that
contain OmegaConf objects and other metadata. Wrap those loads with
:func:`trusted_torch_load_context`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

import torch


@contextmanager
def trusted_torch_load_context() -> Any:
    """Temporarily set ``torch.load`` default to ``weights_only=False`` for trusted files.

    Use only for checkpoints from a trusted source (your own training runs).
    """
    orig: Callable[..., Any] = torch.load

    def _load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return orig(*args, **kwargs)

    torch.load = _load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = orig  # type: ignore[assignment]
