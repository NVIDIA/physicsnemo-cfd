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

"""CI tests for physicsnemo.cfd.evaluation (mesh bridge, metrics, config, engine helpers)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import numpy as np
import pyvista as pv

from physicsnemo.cfd.postprocessing_tools.metrics.aero_forces import (
    compute_drag_and_lift,
)
from physicsnemo.cfd.postprocessing_tools.metrics.l2_errors import compute_l2_errors
from physicsnemo.cfd.evaluation.benchmarks.engine import (
    _call_metric,
    _normalize_metrics_config,
)
from physicsnemo.cfd.evaluation.benchmarks.report_plugins import (
    _apply_default_case_ids_to_visuals,
)
from physicsnemo.cfd.evaluation.benchmarks.engine import (
    _retain_comparison_mesh_for_visual_context,
)
from physicsnemo.cfd.evaluation.config import Config, OutputConfig, ReportsConfig
from physicsnemo.cfd.evaluation.datasets.adapters.drivaerml import (
    DRIVAER_TURBULENT_VISCOSITY_NAMES,
    DRIVAER_VOLUME_VELOCITY_NAMES,
    DrivAerMLAdapter,
)
from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase
from physicsnemo.cfd.evaluation.datasets.vtk_ground_truth import (
    extract_volume_fields_from_mesh,
)
from physicsnemo.cfd.evaluation.metrics import get_metric, list_metrics
from physicsnemo.cfd.evaluation.metrics.builtin.l2 import (
    l2_pressure_surface,
    l2_pressure_volume,
)
from physicsnemo.cfd.evaluation.common.io import surface_polydata_from_case
from physicsnemo.cfd.evaluation.metrics.mesh_bridge import build_comparison_mesh
from physicsnemo.cfd.evaluation.metrics.registry import (
    register_metric,
    unregister_metric,
)

_CI_TEST_LEGACY_METRIC_NAME = "_ci_test_legacy_metric"


@pytest.fixture
def ci_test_legacy_metric() -> Iterator[None]:
    """Register a domain-agnostic legacy metric only for one test and remove after."""

    def legacy(_gt: dict, _pred: dict) -> float:
        """Trivial legacy-style metric returning a constant for fixture wiring."""
        return 1.0

    register_metric(_CI_TEST_LEGACY_METRIC_NAME, legacy)
    yield
    unregister_metric(_CI_TEST_LEGACY_METRIC_NAME)


def test_normalize_metrics_config_strings_and_dicts() -> None:
    """Mixed string/dict metric specs normalize to ``(name, kwargs)`` pairs."""
    specs = _normalize_metrics_config(
        [
            "l2_pressure",
            {"name": "drag", "coeff": 1.5},
        ]
    )
    assert specs == [("l2_pressure", {}), ("drag", {"coeff": 1.5})]


def test_reports_visual_case_ids_and_mesh_retention() -> None:
    """Comparison-mesh retention honors ``visual_case_ids`` and the global ``enabled`` flag."""
    rep_all = ReportsConfig(
        enabled=True, visuals=["field_comparison_surface"], visual_case_ids=None
    )
    assert _retain_comparison_mesh_for_visual_context(rep_all, "run_1") is True
    rep_sub = ReportsConfig(
        enabled=True,
        visuals=["field_comparison_surface"],
        visual_case_ids=["run_1"],
    )
    assert _retain_comparison_mesh_for_visual_context(rep_sub, "run_1") is True
    assert _retain_comparison_mesh_for_visual_context(rep_sub, "run_99") is False
    rep_off = ReportsConfig(
        enabled=False, visuals=["field_comparison_surface"], visual_case_ids=["run_1"]
    )
    assert _retain_comparison_mesh_for_visual_context(rep_off, "run_1") is False


def test_apply_default_case_ids_to_visuals() -> None:
    """Visuals without explicit ``case_ids`` inherit from ``reports.visual_case_ids``."""
    cfg = Config(
        reports=ReportsConfig(visual_case_ids=["a", "b"], visuals=[], enabled=True),
    )
    specs = [
        ("line_plot", {"canonical_key": "pressure"}),
        ("line_plot", {"case_ids": ["c"], "canonical_key": "p"}),
    ]
    out = _apply_default_case_ids_to_visuals(cfg, specs)
    assert out[0][1]["case_ids"] == ["a", "b"]
    assert out[1][1]["case_ids"] == ["c"]
    cfg2 = Config(reports=ReportsConfig(visual_case_ids=None, visuals=[]))
    assert _apply_default_case_ids_to_visuals(cfg2, specs) == specs


def test_config_from_dict_merges_output_and_reports() -> None:
    """``Config.from_dict`` merges ``output`` field-name overrides and ``reports`` settings."""
    cfg = Config.from_dict(
        {
            "metrics": ["l2_pressure"],
            "output": {
                "ground_truth_mesh_field_names": {"pressure": "p_gt_custom"},
            },
            "reports": {
                "enabled": True,
                "plugins": [{"kind": "stub"}],
                "visual_case_ids": ["run_1"],
            },
        }
    )
    assert cfg.output.ground_truth_mesh_field_names["pressure"] == "p_gt_custom"
    assert cfg.reports.enabled is True
    assert cfg.reports.plugins == [{"kind": "stub"}]
    assert cfg.reports.visual_case_ids == ["run_1"]


def test_config_from_dict_surface_interpolate_output_keys() -> None:
    """``output.surface_interpolate_*`` keys round-trip through ``Config.from_dict``."""
    cfg = Config.from_dict(
        {
            "output": {
                "surface_interpolate_point_to_cell_for_metrics": True,
                "surface_metrics_idw_k": 7,
                "surface_metrics_idw_device": "cpu",
            },
        }
    )
    assert cfg.output.surface_interpolate_point_to_cell_for_metrics is True
    assert cfg.output.surface_metrics_idw_k == 7
    assert cfg.output.surface_metrics_idw_device == "cpu"


def test_config_from_dict_model_package_keys() -> None:
    """``model.package`` and ``*_relpath`` keys load and are stripped from merged load kwargs."""
    cfg = Config.from_dict(
        {
            "model": {
                "name": "fignet_surface",
                "package": "hf://nvidia/demo@main",
                "checkpoint_relpath": "ckpt/model.pt",
                "stats_relpath": "global_stats.json",
            },
        }
    )
    assert cfg.model.package == "hf://nvidia/demo@main"
    assert cfg.model.checkpoint_relpath == "ckpt/model.pt"
    assert cfg.model.stats_relpath == "global_stats.json"
    assert "package" not in cfg.model.merged_kwargs_for_load()


def test_list_metrics_includes_core_builtin() -> None:
    """Built-in core metrics (``l2_pressure``, ``drag``) are visible to :func:`list_metrics`."""
    names = list_metrics()
    assert "l2_pressure" in names
    assert "drag" in names


def test_surface_polydata_from_case_skips_pv_read(monkeypatch, tmp_path) -> None:
    """When ``reference_geometry`` is set, do not ``pv.read`` ``mesh_path``."""

    def _fail_read(*args, **kwargs):
        raise AssertionError(f"unexpected pv.read call: args={args!r}")

    monkeypatch.setattr(pv, "read", _fail_read)
    ref = pv.Plane(i_resolution=2, j_resolution=2)
    case = CanonicalCase(
        case_id="run_1",
        mesh_path=str(tmp_path / "missing_boundary.vtp"),
        mesh_type="cell",
        ground_truth=None,
        inference_domain="surface",
        reference_geometry=ref,
    )
    m = surface_polydata_from_case(case)
    assert isinstance(m, pv.PolyData)


def test_build_comparison_mesh_with_case_reference_geometry_skips_pv_read(
    monkeypatch, tmp_path
) -> None:
    """Benchmark path passes ``mesh_override=case.reference_geometry``; parity with explicit override."""

    def _fail_read(*args, **kwargs):
        raise AssertionError("comparison mesh load must not pv.read case.mesh_path")

    monkeypatch.setattr(pv, "read", _fail_read)
    base = pv.Plane(i_resolution=4, j_resolution=4)
    mesh0 = base.point_data_to_cell_data(pass_point_data=True)
    n_cells = mesh0.n_cells
    p = np.ones(n_cells, dtype=np.float64)
    wss = np.zeros((n_cells, 3), dtype=np.float64)
    case = CanonicalCase(
        case_id="syn",
        mesh_path=str(tmp_path / "fake.vtp"),
        mesh_type="cell",
        ground_truth={"pressure": p, "shear_stress": wss},
        inference_domain="surface",
        reference_geometry=mesh0,
    )
    pred = {"pressure": p.copy(), "shear_stress": wss.copy()}
    mesh_out, dtype = build_comparison_mesh(
        case,
        pred,
        OutputConfig(),
        mesh_override=case.reference_geometry,
    )
    assert dtype == "cell"
    assert mesh_out.n_cells == n_cells


def test_build_comparison_mesh_surface_zero_l2_when_identical() -> None:
    """In-memory surface mesh + synthetic numpy fields; no VTP read/write."""
    base = pv.Plane(i_resolution=6, j_resolution=6)
    mesh0 = base.point_data_to_cell_data(pass_point_data=True)
    n_cells = mesh0.n_cells
    p = np.random.randn(n_cells).astype(np.float64)
    wss = np.random.randn(n_cells, 3).astype(np.float64)

    case = CanonicalCase(
        case_id="syn",
        mesh_path="",  # unused when mesh_override is set
        mesh_type="cell",
        ground_truth={"pressure": p, "shear_stress": wss},
        inference_domain="surface",
    )
    pred = {"pressure": p.copy(), "shear_stress": wss.copy()}
    output = OutputConfig()
    mesh, dtype = build_comparison_mesh(case, pred, output, mesh_override=mesh0)
    gtn = output.ground_truth_mesh_field_names["pressure"]
    prn = output.mesh_field_names["pressure"]
    d = compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
    key = f"{gtn}_l2_error"
    assert abs(float(d[key])) < 1e-10

    v = l2_pressure_surface(
        case.ground_truth or {},
        pred,
        case=case,
        comparison_mesh=mesh,
        metric_dtype=dtype,
        output=output,
    )
    assert abs(v) < 1e-10


def test_build_comparison_mesh_volume_zero_l2_when_identical() -> None:
    """In-memory volume grid + synthetic numpy fields; matches surface zero-L2 regression for volume."""
    volume = pv.ImageData(
        dimensions=(6, 5, 4), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
    )
    n_pts = volume.n_points
    p = np.random.randn(n_pts).astype(np.float64)
    vel = np.random.randn(n_pts, 3).astype(np.float64)
    nut = np.random.randn(n_pts).astype(np.float64)

    case = CanonicalCase(
        case_id="syn_vol",
        mesh_path="",
        mesh_type="point",
        ground_truth={
            "pressure": p,
            "velocity": vel,
            "turbulent_viscosity": nut,
        },
        inference_domain="volume",
    )
    pred = {
        "pressure": p.copy(),
        "velocity": vel.copy(),
        "turbulent_viscosity": nut.copy(),
    }
    output = OutputConfig()
    mesh, dtype = build_comparison_mesh(case, pred, output, mesh_override=volume)
    assert dtype == "point"
    gtn = output.ground_truth_volume_mesh_field_names["pressure"]
    prn = output.volume_mesh_field_names["pressure"]
    d = compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
    key = f"{gtn}_l2_error"
    assert abs(float(d[key])) < 1e-10

    v = l2_pressure_volume(
        case.ground_truth or {},
        pred,
        case=case,
        comparison_mesh=mesh,
        metric_dtype=dtype,
        output=output,
    )
    assert abs(v) < 1e-10


def test_build_comparison_mesh_surface_point_dtype_aligns_with_n_points() -> None:
    """Surface ``mesh_type: point`` keeps point dofs (e.g. MeshGraphNet / xmgn_surface + aligned GT)."""
    base = pv.Plane(i_resolution=6, j_resolution=6)
    n_pts = base.n_points
    p = np.random.randn(n_pts).astype(np.float64)
    wss = np.random.randn(n_pts, 3).astype(np.float64)

    case = CanonicalCase(
        case_id="syn_pt",
        mesh_path="",
        mesh_type="point",
        ground_truth={"pressure": p, "shear_stress": wss},
        inference_domain="surface",
    )
    pred = {"pressure": p.copy(), "shear_stress": wss.copy()}
    output = OutputConfig()
    mesh, dtype = build_comparison_mesh(case, pred, output, mesh_override=base)
    assert dtype == "point"
    assert mesh.n_points == n_pts
    gtn = output.ground_truth_mesh_field_names["pressure"]
    prn = output.mesh_field_names["pressure"]
    d = compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
    key = f"{gtn}_l2_error"
    assert abs(float(d[key])) < 1e-10


def test_build_comparison_mesh_surface_point_to_cell_idw_for_metrics() -> None:
    """Point-based surface fields promoted to cells for cell-typed metrics (XmGN / FiGNet pattern)."""
    base = pv.Plane(i_resolution=6, j_resolution=6)
    n_pts = base.n_points
    p = np.ones(n_pts, dtype=np.float64) * 0.5
    wss = np.zeros((n_pts, 3), dtype=np.float64)

    case = CanonicalCase(
        case_id="syn_idw",
        mesh_path="",
        mesh_type="point",
        ground_truth={"pressure": p, "shear_stress": wss},
        inference_domain="surface",
    )
    pred = {"pressure": p.copy(), "shear_stress": wss.copy()}
    output = OutputConfig(
        surface_interpolate_point_to_cell_for_metrics=True,
        surface_metrics_idw_k=5,
        surface_metrics_idw_device="cpu",
    )
    mesh, dtype = build_comparison_mesh(case, pred, output, mesh_override=base)
    assert dtype == "cell"
    gtn = output.ground_truth_mesh_field_names["pressure"]
    prn = output.mesh_field_names["pressure"]
    gtw = output.ground_truth_mesh_field_names["shear_stress"]
    assert gtn in mesh.cell_data and prn in mesh.cell_data
    assert gtn not in mesh.point_data
    d = compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
    assert abs(float(d[f"{gtn}_l2_error"])) < 1e-9

    cd_gt, *_ = compute_drag_and_lift(
        mesh,
        pressure_field=gtn,
        wss_field=gtw,
        dtype="cell",
    )
    assert not np.isnan(cd_gt)


def test_build_comparison_mesh_volume_cell_dtype_matches_n_cells() -> None:
    """Volume VTU-style fields on cells (DrivAerML default): lengths match ``n_cells``, not ``n_points``."""
    grid = pv.ImageData(dimensions=(5, 6, 7))
    nc, np_ = grid.n_cells, grid.n_points
    assert nc != np_
    pr = np.random.randn(nc).astype(np.float64)
    vel = np.random.randn(nc, 3).astype(np.float64)
    nut = np.random.randn(nc).astype(np.float64)

    case = CanonicalCase(
        case_id="vol_cell",
        mesh_path="",
        mesh_type="cell",
        ground_truth={"pressure": pr, "velocity": vel, "turbulent_viscosity": nut},
        inference_domain="volume",
    )
    pred = {
        "pressure": pr.copy(),
        "velocity": vel.copy(),
        "turbulent_viscosity": nut.copy(),
    }
    output = OutputConfig()
    mesh, dtype = build_comparison_mesh(case, pred, output, mesh_override=grid)
    assert dtype == "cell"
    gtn = output.ground_truth_volume_mesh_field_names["pressure"]
    prn = output.volume_mesh_field_names["pressure"]
    d = compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
    assert abs(float(d[f"{gtn}_l2_error"])) < 1e-10


def test_build_comparison_mesh_volume_point_dtype_matches_n_points() -> None:
    """Point-centered volume fields (``mesh_type: point`` or array length == ``n_points``)."""
    grid = pv.ImageData(dimensions=(5, 6, 7))
    nc, np_ = grid.n_cells, grid.n_points
    assert nc != np_
    pr = np.random.randn(np_).astype(np.float64)
    vel = np.random.randn(np_, 3).astype(np.float64)
    nut = np.random.randn(np_).astype(np.float64)

    case = CanonicalCase(
        case_id="vol_pt",
        mesh_path="",
        mesh_type="point",
        ground_truth={"pressure": pr, "velocity": vel, "turbulent_viscosity": nut},
        inference_domain="volume",
    )
    pred = {
        "pressure": pr.copy(),
        "velocity": vel.copy(),
        "turbulent_viscosity": nut.copy(),
    }
    output = OutputConfig()
    mesh, dtype = build_comparison_mesh(case, pred, output, mesh_override=grid)
    assert dtype == "point"
    gtn = output.ground_truth_volume_mesh_field_names["pressure"]
    prn = output.volume_mesh_field_names["pressure"]
    d = compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
    assert abs(float(d[f"{gtn}_l2_error"])) < 1e-10


def test_build_comparison_mesh_volume_raises_when_reference_length_matches_neither_points_nor_cells() -> (
    None
):
    """Fail fast when volume GT/pred length disagrees with topology (mirrors surface mesh_bridge)."""
    grid = pv.ImageData(dimensions=(5, 6, 7))
    nc, np_ = grid.n_cells, grid.n_points
    assert nc != np_
    bad_n = 1
    pr = np.random.randn(bad_n).astype(np.float64)
    vel = np.random.randn(bad_n, 3).astype(np.float64)
    nut = np.random.randn(bad_n).astype(np.float64)
    case = CanonicalCase(
        case_id="vol_bad_dof",
        mesh_path="",
        mesh_type="cell",
        ground_truth={"pressure": pr, "velocity": vel, "turbulent_viscosity": nut},
        inference_domain="volume",
    )
    pred = {
        "pressure": pr.copy(),
        "velocity": vel.copy(),
        "turbulent_viscosity": nut.copy(),
    }
    output = OutputConfig()
    with pytest.raises(ValueError, match="volume field sample count"):
        build_comparison_mesh(case, pred, output, mesh_override=grid)


def test_extract_volume_fields_generic_uses_vanilla_cfd_names() -> None:
    """Generic extractor defaults are dataset-agnostic (``UMean`` / ``nutMean`` / ``pMean``)."""
    grid = pv.ImageData(dimensions=(4, 4, 4))
    n = grid.n_cells
    rng = np.random.default_rng(0)
    grid.cell_data["pMean"] = rng.standard_normal(n).astype(np.float32)
    grid.cell_data["UMean"] = rng.standard_normal((n, 3)).astype(np.float32)
    grid.cell_data["nutMean"] = rng.standard_normal(n).astype(np.float32)

    gt, loc = extract_volume_fields_from_mesh(grid, data_type="auto")
    assert loc == "cell"
    assert gt is not None
    assert set(gt.keys()) == {"pressure", "velocity", "turbulent_viscosity"}


def test_drivaerml_adapter_finds_trim_only_volume_fields(tmp_path) -> None:
    """DrivAer adapter ships ``*MeanTrim`` defaults so Trim-only VTUs populate velocity / νₜ
    without requiring per-config ``velocity_field_names`` / ``turbulent_viscosity_field_names``.
    """
    assert "UMeanTrim" in DRIVAER_VOLUME_VELOCITY_NAMES
    assert "nutMeanTrim" in DRIVAER_TURBULENT_VISCOSITY_NAMES

    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    grid = pv.ImageData(dimensions=(4, 4, 4))
    n = grid.n_cells
    rng = np.random.default_rng(0)
    grid.cell_data["pMeanTrim"] = rng.standard_normal(n).astype(np.float32)
    grid.cell_data["UMeanTrim"] = rng.standard_normal((n, 3)).astype(np.float32)
    grid.cell_data["nutMeanTrim"] = rng.standard_normal(n).astype(np.float32)
    ugrid = grid.cast_to_unstructured_grid()
    ugrid.save(str(run_dir / "volume_1.vtu"))

    adapter = DrivAerMLAdapter(root=str(tmp_path), inference_domain="volume")
    case = adapter.load_case("run_1")
    assert case.inference_domain == "volume"
    assert case.ground_truth is not None
    assert set(case.ground_truth.keys()) == {
        "pressure",
        "velocity",
        "turbulent_viscosity",
    }
    assert case.ground_truth["velocity"].shape == (n, 3)
    assert case.ground_truth["turbulent_viscosity"].shape == (n,)


def test_legacy_metric_call_without_extended_kwargs(
    ci_test_legacy_metric: None,
) -> None:
    """Metrics with a fixed (gt, pred) signature fall back when extended kwargs are rejected."""
    fn = get_metric(_CI_TEST_LEGACY_METRIC_NAME)
    out = _call_metric(
        fn,
        {},
        {},
        case=None,
        comparison_mesh=None,
        metric_dtype=None,
        output=OutputConfig(),
        mkwargs={},
    )
    assert out == 1.0
