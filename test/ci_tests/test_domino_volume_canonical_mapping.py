# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards on ``domino_volume_predictions_to_canonical`` channel layout and canonical key uniqueness."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from physicsnemo.cfd.evaluation.models.wrappers.domino.inference import (
    domino_volume_predictions_to_canonical,
)


def _vol_cfg(solution: dict[str, str]) -> OmegaConf:
    """Build a minimal OmegaConf config with the given ``variables.volume.solution`` mapping."""
    return OmegaConf.create({"variables": {"volume": {"solution": solution}}})


def test_domino_volume_predictions_channel_count_mismatch_raises() -> None:
    """Summed width from ``variables.volume.solution`` must match the tensor last dimension."""
    cfg = _vol_cfg({"u": "vector", "p": "scalar"})  # 4 channels
    pred = torch.zeros(8, 5)
    with pytest.raises(ValueError, match="volume solution channels"):
        domino_volume_predictions_to_canonical(pred, cfg)


def test_domino_volume_predictions_duplicate_canonical_pressure_raises() -> None:
    """Substring heuristics must not map two fields to the same canonical key."""
    cfg = _vol_cfg(
        {
            "velocity": "vector",
            "p_wall": "scalar",
            "p_corner": "scalar",
        }
    )
    pred = torch.zeros(8, 5)
    with pytest.raises(ValueError, match="same canonical output 'pressure'"):
        domino_volume_predictions_to_canonical(pred, cfg)


def test_domino_volume_predictions_unknown_vector_field_raises() -> None:
    """Non-velocity vector channels have no physical scaling — misconfiguration must not pass silently."""
    cfg = _vol_cfg({"vorticity": "vector", "p": "scalar", "nut": "scalar"})
    pred = torch.zeros(6, 5)
    with pytest.raises(ValueError, match="not classified as velocity"):
        domino_volume_predictions_to_canonical(pred, cfg)


def test_domino_volume_predictions_drivaer_like_mapping_ok() -> None:
    """A DrivAer-like config maps to canonical velocity/pressure/turbulent_viscosity outputs."""
    cfg = _vol_cfg(
        {
            "velocity": "vector",
            "p": "scalar",
            "nut": "scalar",
        }
    )
    pred = torch.zeros(12, 5)
    out = domino_volume_predictions_to_canonical(pred, cfg)
    assert out["velocity"].shape == (12, 3)
    assert out["pressure"].shape == (12,)
    assert out["turbulent_viscosity"].shape == (12,)
