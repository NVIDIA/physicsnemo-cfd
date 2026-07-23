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

"""DrivAerStar adapter regression tests: the WSS sign-convention decision, pressure renaming,
normals/area removal, and the integrated ground-truth sign (drag/lift depend on it)."""

from __future__ import annotations

import numpy as np
import pyvista as pv
import pytest

from physicsnemo.cfd.evaluation.datasets.adapters.drivaerstar import (
    DEFAULT_PRESSURE_OUT_NAME,
    DEFAULT_SHEAR_OUT_NAME,
    DrivAerStarAdapter,
)


def _plane_with_fields() -> pv.PolyData:
    """A small plane carrying DrivAerStar-style cell arrays (constant WSS for sign checks)."""
    mesh = pv.Plane(i_resolution=3, j_resolution=3)
    n = mesh.n_cells
    mesh.cell_data["Pressure"] = np.arange(n, dtype=np.float64)
    mesh.cell_data["WallShearStressi"] = np.full(n, 1.0, dtype=np.float64)
    mesh.cell_data["WallShearStressj"] = np.full(n, 2.0, dtype=np.float64)
    mesh.cell_data["WallShearStressk"] = np.full(n, 3.0, dtype=np.float64)
    mesh.cell_data["Normals"] = np.zeros((n, 3), dtype=np.float64)
    mesh.cell_data["Area"] = np.ones(n, dtype=np.float64)
    return mesh


def test_flip_wss_sign_true_negates_components(tmp_path) -> None:
    """flip_wss_sign=True (default) negates the combined WSS to the DrivAerML convention."""
    adapter = DrivAerStarAdapter(root=str(tmp_path), flip_wss_sign=True)
    mesh = _plane_with_fields()
    adapter._combine_wss(mesh, "1")
    wss = np.asarray(mesh.cell_data[DEFAULT_SHEAR_OUT_NAME])
    assert wss.shape[1] == 3
    assert np.allclose(wss[:, 0], -1.0)
    assert np.allclose(wss[:, 1], -2.0)
    assert np.allclose(wss[:, 2], -3.0)
    # Source component arrays are consumed into the combined vector.
    assert "WallShearStressi" not in mesh.cell_data


def test_flip_wss_sign_false_keeps_native_sign(tmp_path) -> None:
    """flip_wss_sign=False keeps the native DrivAerStar sign (as the UQ example config sets)."""
    adapter = DrivAerStarAdapter(root=str(tmp_path), flip_wss_sign=False)
    mesh = _plane_with_fields()
    adapter._combine_wss(mesh, "1")
    wss = np.asarray(mesh.cell_data[DEFAULT_SHEAR_OUT_NAME])
    assert np.allclose(wss[:, 0], 1.0)
    assert np.allclose(wss[:, 1], 2.0)
    assert np.allclose(wss[:, 2], 3.0)


def test_combine_wss_missing_components_raises_loudly(tmp_path) -> None:
    """A missing/misnamed WSS component fails loudly (not a silent no-op) during preparation."""
    adapter = DrivAerStarAdapter(root=str(tmp_path))
    with pytest.raises(ValueError):
        adapter._combine_wss(pv.Plane(i_resolution=2, j_resolution=2), "1")


def test_rename_pressure_and_missing_raises(tmp_path) -> None:
    """Pressure is renamed to the DrivAerML name; a missing pressure array raises loudly."""
    adapter = DrivAerStarAdapter(root=str(tmp_path))
    mesh = _plane_with_fields()
    adapter._rename_pressure(mesh, "1")
    assert DEFAULT_PRESSURE_OUT_NAME in mesh.cell_data
    assert "Pressure" not in mesh.cell_data
    with pytest.raises(ValueError):
        adapter._rename_pressure(pv.Plane(i_resolution=2, j_resolution=2), "1")


def test_drop_normals_area_removes_explicit_arrays() -> None:
    """Explicit Normals/Area arrays are dropped so forces recompute from mesh topology."""
    mesh = _plane_with_fields()
    DrivAerStarAdapter._drop_normals_area(mesh)
    assert "Normals" not in mesh.cell_data and "Area" not in mesh.cell_data
    assert "Normals" not in mesh.point_data and "Area" not in mesh.point_data


def test_load_case_ground_truth_respects_wss_sign(tmp_path) -> None:
    """End-to-end load_case exposes ground-truth WSS in the configured sign (drives drag/lift)."""
    _plane_with_fields().save(str(tmp_path / "1.vtk"))
    adapter = DrivAerStarAdapter(root=str(tmp_path), flip_wss_sign=True)
    case = adapter.load_case("1")
    assert case.ground_truth is not None
    shear = np.asarray(case.ground_truth["shear_stress"])
    assert np.allclose(shear[:, 0], -1.0)
    assert np.allclose(shear[:, 1], -2.0)
    assert np.allclose(shear[:, 2], -3.0)
