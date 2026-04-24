# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI tests for physicsnemo.cfd.evaluation (mesh bridge, metrics, config, engine helpers)."""

from __future__ import annotations

import importlib.util

import pytest

# Bench L2 imports physicsnemo.utils.sdf at module level; skip whole module if missing.
if importlib.util.find_spec("physicsnemo.utils.sdf") is None:
    pytest.skip(
        "physicsnemo.utils.sdf not available (install/upgrade nvidia-physicsnemo)",
        allow_module_level=True,
    )

import numpy as np
import pyvista as pv

from physicsnemo.cfd.postprocessing_tools.metrics.aero_forces import compute_drag_and_lift
from physicsnemo.cfd.postprocessing_tools.metrics.l2_errors import compute_l2_errors
from physicsnemo.cfd.evaluation.benchmarks.engine import _call_metric, _normalize_metrics_config
from physicsnemo.cfd.evaluation.benchmarks.report_plugins import _apply_default_case_ids_to_visuals
from physicsnemo.cfd.evaluation.benchmarks.engine import _retain_comparison_mesh_for_visual_context
from physicsnemo.cfd.evaluation.config import Config, OutputConfig, ReportsConfig
from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase
from physicsnemo.cfd.evaluation.metrics import get_metric, list_metrics
from physicsnemo.cfd.evaluation.metrics.builtin.l2 import l2_pressure
from physicsnemo.cfd.evaluation.metrics.mesh_bridge import build_comparison_mesh
from physicsnemo.cfd.evaluation.metrics.registry import register_metric


def test_normalize_metrics_config_strings_and_dicts() -> None:
    specs = _normalize_metrics_config(
        [
            "l2_pressure",
            {"name": "drag", "coeff": 1.5},
        ]
    )
    assert specs == [("l2_pressure", {}), ("drag", {"coeff": 1.5})]


def test_reports_visual_case_ids_and_mesh_retention() -> None:
    rep_all = ReportsConfig(enabled=True, visuals=["field_comparison_surface"], visual_case_ids=None)
    assert _retain_comparison_mesh_for_visual_context(rep_all, "run_1") is True
    rep_sub = ReportsConfig(
        enabled=True,
        visuals=["field_comparison_surface"],
        visual_case_ids=["run_1"],
    )
    assert _retain_comparison_mesh_for_visual_context(rep_sub, "run_1") is True
    assert _retain_comparison_mesh_for_visual_context(rep_sub, "run_99") is False
    rep_off = ReportsConfig(enabled=False, visuals=["field_comparison_surface"], visual_case_ids=["run_1"])
    assert _retain_comparison_mesh_for_visual_context(rep_off, "run_1") is False


def test_apply_default_case_ids_to_visuals() -> None:
    cfg = Config(
        reports=ReportsConfig(visual_case_ids=["a", "b"], visuals=[], enabled=True),
    )
    specs = [("line_plot", {"canonical_key": "pressure"}), ("line_plot", {"case_ids": ["c"], "canonical_key": "p"})]
    out = _apply_default_case_ids_to_visuals(cfg, specs)
    assert out[0][1]["case_ids"] == ["a", "b"]
    assert out[1][1]["case_ids"] == ["c"]
    cfg2 = Config(reports=ReportsConfig(visual_case_ids=None, visuals=[]))
    assert _apply_default_case_ids_to_visuals(cfg2, specs) == specs


def test_config_from_dict_merges_output_and_reports() -> None:
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
    cfg = Config.from_dict(
        {
            "model": {
                "name": "fignet",
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
    names = list_metrics()
    assert "l2_pressure" in names
    assert "drag" in names


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

    v = l2_pressure(
        case.ground_truth or {},
        pred,
        case=case,
        comparison_mesh=mesh,
        metric_dtype=dtype,
        output=output,
    )
    assert abs(v) < 1e-10


def test_build_comparison_mesh_surface_point_dtype_aligns_with_n_points() -> None:
    """Surface ``mesh_type: point`` keeps point dofs (e.g. MeshGraphNet / xmgn + aligned GT)."""
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


def test_legacy_metric_call_without_extended_kwargs() -> None:
    """Metrics with a fixed (gt, pred) signature fall back when extended kwargs are rejected."""

    def legacy(gt: dict, pred: dict) -> float:
        return 1.0

    register_metric("_ci_test_legacy_metric", legacy)
    fn = get_metric("_ci_test_legacy_metric")
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
