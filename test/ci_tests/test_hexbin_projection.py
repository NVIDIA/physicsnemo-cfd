# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``plot_projections_hexbin`` world-plane indexing."""

from __future__ import annotations

import pytest

from physicsnemo.cfd.postprocessing_tools.visualization import utils as viz_utils


def test_parse_hexbin_direction_basic() -> None:
    assert viz_utils._parse_hexbin_direction("XY") == ("XY", False)
    assert viz_utils._parse_hexbin_direction("-YZ") == ("YZ", True)
    assert viz_utils._parse_hexbin_direction("xz") == ("XZ", False)


def test_parse_hexbin_direction_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown plot_projections_hexbin"):
        viz_utils._parse_hexbin_direction("XYZZ")


@pytest.mark.parametrize(
    "plane,expected",
    [
        ("XY", (0, 1)),
        ("YZ", (1, 2)),
        ("ZX", (2, 0)),
        ("XZ", (0, 2)),
    ],
)
def test_world_indices_for_plane(plane: str, expected: tuple[int, int]) -> None:
    assert viz_utils._world_indices_for_plane(plane) == expected


@pytest.mark.parametrize(
    "plane,is_neg,inv_x,inv_y",
    [
        ("XY", False, False, False),
        ("YZ", False, False, False),
        ("ZX", False, False, False),
        ("XZ", False, False, False),
        ("XY", True, True, False),
        ("YZ", True, False, True),
        ("ZX", True, False, True),
        ("XZ", True, False, True),
    ],
)
def test_matplotlib_inverts(plane: str, is_neg: bool, inv_x: bool, inv_y: bool) -> None:
    assert viz_utils._matplotlib_inverts(plane, is_neg) == (inv_x, inv_y)
