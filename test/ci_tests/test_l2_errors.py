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

"""CI tests for ``physicsnemo.cfd.postprocessing_tools.metrics.l2_errors``.

Covers the shared helpers (``_dof_coordinates``, ``_bounds_mask``) and exercises
both ``compute_l2_errors`` and ``compute_error_vs_sdf`` with ``dtype="cell"``
and a non-trivial ``bounds`` filter.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import pytest

from physicsnemo.cfd.postprocessing_tools.metrics.l2_errors import (
    _bounds_mask,
    _classify_fields,
    _dof_coordinates,
    compute_error_vs_sdf,
    compute_l2_errors,
)


def _grid(
    dimensions: tuple[int, int, int] = (5, 6, 7), spacing: float = 0.2
) -> pv.ImageData:
    """ImageData grid centered on the origin so the small-sphere SDF straddles zero."""
    origin: tuple[float, float, float] = (
        -0.5 * (dimensions[0] - 1) * spacing,
        -0.5 * (dimensions[1] - 1) * spacing,
        -0.5 * (dimensions[2] - 1) * spacing,
    )
    return pv.ImageData(
        dimensions=dimensions,
        spacing=(spacing, spacing, spacing),
        origin=origin,
    )


def _small_sphere_stl() -> pv.PolyData:
    """Tiny triangulated sphere used as the SDF surface (~80 triangles → CPU-fast)."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8, radius=0.3).triangulate()


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def test_dof_coordinates_point_returns_node_coords() -> None:
    """``dtype="point"`` → ``data.points`` (length ``n_points``)."""
    grid = _grid()
    coords = _dof_coordinates(grid, "point")
    assert coords.shape == (grid.n_points, 3)
    assert np.allclose(coords, np.asarray(grid.points))


def test_dof_coordinates_cell_returns_cell_centers() -> None:
    """``dtype="cell"`` → ``cell_centers().points`` (length ``n_cells``); critical for masks."""
    grid = _grid()
    assert grid.n_points != grid.n_cells  # guard the test premise
    coords = _dof_coordinates(grid, "cell")
    assert coords.shape == (grid.n_cells, 3)
    assert np.allclose(coords, np.asarray(grid.cell_centers().points))


def test_bounds_mask_none_short_circuits() -> None:
    """A None bounds spec short-circuits to no mask."""
    coords = np.zeros((5, 3))
    assert _bounds_mask(coords, None) is None


def test_bounds_mask_inclusive_aabb() -> None:
    """The bounds mask is an inclusive axis-aligned box test."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [-0.1, 0.5, 0.5],
            [0.5, 1.1, 0.5],
        ]
    )
    mask = _bounds_mask(coords, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    assert mask is not None
    assert mask.tolist() == [True, True, True, False, False]


def test_classify_fields_scalar_vs_vector() -> None:
    """Field classification distinguishes scalar from vector arrays by component count."""
    grid = _grid()
    grid.point_data["scalar"] = np.zeros(grid.n_points, dtype=np.float64)
    grid.point_data["vector"] = np.zeros((grid.n_points, 3), dtype=np.float64)
    types = _classify_fields(grid, ["scalar", "vector"], dtype="point")
    assert types == {"scalar": "scalar", "vector": "vector"}


# ---------------------------------------------------------------------------
# compute_l2_errors — cell + bounds regression (no SDF / warp dependency)
# ---------------------------------------------------------------------------


def test_compute_l2_errors_cell_with_full_bounds_matches_no_bounds() -> None:
    """Cell dtype + bounds covering the whole AABB must equal the no-bounds result.

    Pre-fix the mask was built from ``data.points`` (length ``n_points``), then
    used to slice the cell-length field array — for ``n_points > n_cells`` this
    raised ``IndexError``; for ``n_points <= n_cells`` it silently truncated
    rows. With the helper using ``cell_centers().points`` the two paths now
    agree.
    """
    grid = _grid(dimensions=(5, 6, 7))
    assert grid.n_points > grid.n_cells
    rng = np.random.default_rng(0)
    grid.cell_data["true"] = rng.standard_normal(grid.n_cells).astype(np.float64)
    grid.cell_data["pred"] = grid.cell_data["true"] + 0.25 * rng.standard_normal(
        grid.n_cells
    ).astype(np.float64)

    no_bounds = compute_l2_errors(grid, ["true"], ["pred"], dtype="cell")
    full_bounds = compute_l2_errors(
        grid,
        ["true"],
        ["pred"],
        bounds=list[float](grid.bounds),
        dtype="cell",
    )
    assert np.isclose(
        no_bounds["true_l2_error"], full_bounds["true_l2_error"], rtol=0, atol=1e-12
    )


def test_compute_l2_errors_cell_with_bounds_zero_when_identical() -> None:
    """Identical fields restricted to a half-domain must yield zero relative L2."""
    grid = _grid(dimensions=(5, 6, 7))
    grid.cell_data["true"] = np.linspace(0.5, 1.5, grid.n_cells, dtype=np.float64)
    grid.cell_data["pred"] = grid.cell_data["true"].copy()

    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    half = [xmin, 0.5 * (xmin + xmax), ymin, ymax, zmin, zmax]
    out = compute_l2_errors(grid, ["true"], ["pred"], bounds=half, dtype="cell")
    assert abs(out["true_l2_error"]) < 1e-12


def test_compute_l2_errors_cell_bounds_actually_filters() -> None:
    """Bounds must change the result when error magnitude varies across the grid.

    Pre-fix the bug was *silent* for ``n_points <= n_cells``: the points-length
    mask just truncated. Use a tiny grid where ``n_points <= n_cells`` is
    impossible (it never is for ``ImageData``), but verify positively that the
    bounded result differs from the unbounded one when error is non-uniform.
    """
    grid = _grid(dimensions=(5, 6, 7))
    centers = np.asarray(grid.cell_centers().points)
    grid.cell_data["true"] = np.zeros(grid.n_cells, dtype=np.float64)
    # Error magnitude grows with x → bounding to negative x must lower L2.
    grid.cell_data["pred"] = (centers[:, 0] - centers[:, 0].min()).astype(np.float64)

    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    neg_x_only = [xmin, 0.0, ymin, ymax, zmin, zmax]
    bounded = compute_l2_errors(
        grid, ["true"], ["pred"], bounds=neg_x_only, dtype="cell"
    )
    unbounded = compute_l2_errors(grid, ["true"], ["pred"], dtype="cell")
    # ``true`` is identically zero so the relative-L2 falls back to ``||t-p||``;
    # restricting to negative x strictly drops large-error points.
    assert bounded["true_l2_error"] < unbounded["true_l2_error"]


# ---------------------------------------------------------------------------
# compute_error_vs_sdf
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sdf_runtime() -> None:
    """Skip ``compute_error_vs_sdf`` tests cleanly when warp / SDF is unavailable."""
    pytest.importorskip("warp")


def _bin_edges() -> np.ndarray:
    """SDF bin edges that straddle the small sphere surface (radius 0.3)."""
    return np.linspace(-0.6, 0.6, 7)


def test_compute_error_vs_sdf_point_zero_when_identical(sdf_runtime: None) -> None:
    """Identical truth/pred → every populated bin has zero mean error."""
    grid = _grid(dimensions=(4, 4, 4))
    grid.point_data["t"] = np.arange(grid.n_points, dtype=np.float64)
    grid.point_data["p"] = grid.point_data["t"].copy()
    out = compute_error_vs_sdf(
        grid,
        ["t"],
        ["p"],
        _small_sphere_stl(),
        _bin_edges(),
        dtype="point",
    )
    errs = out["t_l2_error_histogram"]["mean_errors"]
    assert len(errs) == _bin_edges().size - 1
    for e in errs:
        assert np.isnan(e) or float(e) == 0.0


def test_compute_error_vs_sdf_cell_zero_when_identical(sdf_runtime: None) -> None:
    """Cell-dtype regression: SDF query coordinates must align with cell-length fields.

    Pre-fix the SDF was computed at ``data.points`` (length ``n_points``) while
    ``per_point_error`` was cell-length, so ``per_point_error[mask]`` either
    raised or silently mismatched. With ``_dof_coordinates(... "cell")`` both
    arrays now have length ``n_cells``.
    """
    grid = _grid(dimensions=(4, 4, 4))
    grid.cell_data["t"] = np.arange(grid.n_cells, dtype=np.float64)
    grid.cell_data["p"] = grid.cell_data["t"].copy()
    out = compute_error_vs_sdf(
        grid,
        ["t"],
        ["p"],
        _small_sphere_stl(),
        _bin_edges(),
        dtype="cell",
    )
    errs = out["t_l2_error_histogram"]["mean_errors"]
    assert len(errs) == _bin_edges().size - 1
    for e in errs:
        assert np.isnan(e) or float(e) == 0.0


def test_compute_error_vs_sdf_cell_with_bounds_zero_when_identical(
    sdf_runtime: None,
) -> None:
    """Combined regression: cell dtype *and* bounds — both masks were misaligned before."""
    grid = _grid(dimensions=(4, 4, 4))
    grid.cell_data["t"] = np.arange(grid.n_cells, dtype=np.float64)
    grid.cell_data["p"] = grid.cell_data["t"].copy()
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    half = [xmin, 0.5 * (xmin + xmax), ymin, ymax, zmin, zmax]
    out = compute_error_vs_sdf(
        grid,
        ["t"],
        ["p"],
        _small_sphere_stl(),
        _bin_edges(),
        bounds=half,
        dtype="cell",
    )
    errs = out["t_l2_error_histogram"]["mean_errors"]
    assert len(errs) == _bin_edges().size - 1
    for e in errs:
        assert np.isnan(e) or float(e) == 0.0


def test_compute_error_vs_sdf_vector_field_uses_norm(sdf_runtime: None) -> None:
    """Vector fields contribute ``np.linalg.norm(t - p)`` per DOF; identical → zero."""
    grid = _grid(dimensions=(4, 4, 4))
    grid.point_data["v_t"] = np.arange(grid.n_points * 3, dtype=np.float64).reshape(
        -1, 3
    )
    grid.point_data["v_p"] = grid.point_data["v_t"].copy()
    out = compute_error_vs_sdf(
        grid,
        ["v_t"],
        ["v_p"],
        _small_sphere_stl(),
        _bin_edges(),
        dtype="point",
    )
    errs = out["v_t_l2_error_histogram"]["mean_errors"]
    for e in errs:
        assert np.isnan(e) or float(e) == 0.0


def test_compute_error_vs_sdf_returns_bin_edges_unchanged(sdf_runtime: None) -> None:
    """``bin_edges`` round-trip into the output dict (callers plot against them)."""
    grid = _grid(dimensions=(4, 4, 4))
    grid.point_data["t"] = np.zeros(grid.n_points, dtype=np.float64)
    grid.point_data["p"] = np.zeros(grid.n_points, dtype=np.float64)
    edges = _bin_edges()
    out = compute_error_vs_sdf(
        grid,
        ["t"],
        ["p"],
        _small_sphere_stl(),
        edges,
        dtype="point",
    )
    assert np.allclose(out["t_l2_error_histogram"]["bin_edges"], edges)


def test_compute_error_vs_sdf_per_bin_means_nonnegative(sdf_runtime: None) -> None:
    """Smoke test for non-trivial errors: each populated bin's mean is finite and >= 0."""
    grid = _grid(dimensions=(4, 4, 4))
    pts = np.asarray(grid.points)
    grid.point_data["t"] = np.zeros(grid.n_points, dtype=np.float64)
    grid.point_data["p"] = np.abs(pts[:, 0]).astype(np.float64)
    out = compute_error_vs_sdf(
        grid,
        ["t"],
        ["p"],
        _small_sphere_stl(),
        _bin_edges(),
        dtype="point",
    )
    errs = out["t_l2_error_histogram"]["mean_errors"]
    populated = [float(e) for e in errs if not np.isnan(e)]
    assert populated, "expected at least one populated bin"
    for e in populated:
        assert np.isfinite(e) and e >= 0.0


def test_compute_error_vs_sdf_default_device_is_cpu(sdf_runtime: None) -> None:
    """Default ``device="cpu"`` keeps the function pure on CUDA-less hosts.

    Mirrors the convention used by ``interpolate_mesh_to_pc`` and avoids the
    silent ``cuda:0`` landing on multi-GPU hosts that the reviewer flagged.
    """
    grid = _grid(dimensions=(4, 4, 4))
    grid.point_data["t"] = np.zeros(grid.n_points, dtype=np.float64)
    grid.point_data["p"] = np.zeros(grid.n_points, dtype=np.float64)
    # No ``device=`` arg → CPU; should run without CUDA being available.
    out = compute_error_vs_sdf(
        grid,
        ["t"],
        ["p"],
        _small_sphere_stl(),
        _bin_edges(),
        dtype="point",
    )
    assert "t_l2_error_histogram" in out


def test_compute_error_vs_sdf_legacy_gpu_alias_matches_cuda(sdf_runtime: None) -> None:
    """``device="gpu"`` is the legacy spelling and must produce the same result as ``"cuda"``.

    Skipped when no CUDA device is available so the test still runs on
    GPU-less CI runners.
    """
    import torch as _torch

    if not _torch.cuda.is_available():
        pytest.skip("CUDA unavailable; back-compat alias check requires a GPU")
    grid = _grid(dimensions=(4, 4, 4))
    rng = np.random.default_rng(0)
    grid.point_data["t"] = rng.standard_normal(grid.n_points).astype(np.float64)
    grid.point_data["p"] = grid.point_data["t"] + 0.1 * rng.standard_normal(
        grid.n_points
    ).astype(np.float64)
    out_cuda = compute_error_vs_sdf(
        grid,
        ["t"],
        ["p"],
        _small_sphere_stl(),
        _bin_edges(),
        dtype="point",
        device="cuda",
    )
    out_gpu = compute_error_vs_sdf(
        grid,
        ["t"],
        ["p"],
        _small_sphere_stl(),
        _bin_edges(),
        dtype="point",
        device="gpu",
    )
    cuda_errs = out_cuda["t_l2_error_histogram"]["mean_errors"]
    gpu_errs = out_gpu["t_l2_error_histogram"]["mean_errors"]
    for a, b in zip(cuda_errs, gpu_errs):
        if np.isnan(a) and np.isnan(b):
            continue
        assert np.isclose(a, b, rtol=0, atol=1e-12)
