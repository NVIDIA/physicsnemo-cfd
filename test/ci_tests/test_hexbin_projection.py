# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``plot_projections_hexbin`` world-plane indexing."""

from __future__ import annotations

import pytest

from physicsnemo.cfd.postprocessing_tools.visualization import utils as viz_utils


def test_parse_hexbin_direction_basic() -> None:
    """Direction string parses into ``(plane, is_neg)`` and is case-insensitive."""
    assert viz_utils._parse_hexbin_direction("XY") == ("XY", False)
    assert viz_utils._parse_hexbin_direction("-YZ") == ("YZ", True)
    assert viz_utils._parse_hexbin_direction("xz") == ("XZ", False)


def test_parse_hexbin_direction_rejects_unknown() -> None:
    """Unknown direction strings raise ``ValueError``."""
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
    """Each plane name maps to the expected ``(x_idx, y_idx)`` world coordinate indices."""
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
    """Negative-direction planes invert the matplotlib axis as expected."""
    assert viz_utils._matplotlib_inverts(plane, is_neg) == (inv_x, inv_y)
