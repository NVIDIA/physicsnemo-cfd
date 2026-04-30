# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for aggregate volume report helpers."""

from __future__ import annotations

import pytest

import pyvista as pv

from physicsnemo.cfd.evaluation.reports.builtin.aggregate_volume import (
    _aggregate_filename_stem,
    _axis_linspace,
    _build_structured_grid,
    _union_axis_aligned_bounds,
)


def test_aggregate_filename_stem_sanitized() -> None:
    assert (
        _aggregate_filename_stem("geo/trans", "data:set")
        == "geo_trans_data_set_aggregate"
    )


def test_union_axis_aligned_bounds_two_meshes() -> None:
    a = pv.Sphere(center=(0.0, 0.0, 0.0), radius=1.0)
    b = pv.Sphere(center=(3.0, 0.0, 0.0), radius=1.0)
    u = _union_axis_aligned_bounds([a, b])
    assert len(u) == 6
    assert u[0] == pytest.approx(-1.0, rel=0, abs=0.05)
    assert u[1] == pytest.approx(4.0, rel=0, abs=0.05)


def test_axis_linspace_known_count_and_endpoints() -> None:
    x = _axis_linspace(-3.5, 8.5, 0.03)
    expected_n = max(2, int(round(12.0 / 0.03)) + 1)
    assert len(x) == expected_n == 401
    assert float(x[0]) == pytest.approx(-3.5)
    assert float(x[-1]) == pytest.approx(8.5)


def test_axis_linspace_zero_extent() -> None:
    x = _axis_linspace(1.0, 1.0, 0.03)
    assert len(x) == 1 and float(x[0]) == pytest.approx(1.0)


def test_axis_linspace_rejects_nonpositive_spacing() -> None:
    with pytest.raises(ValueError, match="positive"):
        _axis_linspace(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        _axis_linspace(0.0, 1.0, -0.1)


def test_build_structured_grid_point_count_stable_for_near_identical_bounds() -> None:
    voxel = 0.1
    b1 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    b2 = [0.0, 1.0 + 1e-15, 0.0, 1.0 + 1e-15, 0.0, 1.0 + 1e-15]
    g1 = _build_structured_grid(b1, voxel)
    g2 = _build_structured_grid(b2, voxel)
    assert g1.n_points == g2.n_points
