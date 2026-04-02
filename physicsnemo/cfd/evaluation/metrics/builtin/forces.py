# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Force coefficient metrics via physicsnemo.cfd.bench.metrics.aero_forces."""

from __future__ import annotations

from typing import Any

from physicsnemo.cfd.bench.metric_registry import register_metric
from physicsnemo.cfd.bench.metrics.aero_forces import compute_drag_and_lift
from physicsnemo.cfd.evaluation.metrics.mesh_bridge import build_comparison_mesh


def _resolve_mesh(
    predictions: dict[str, Any],
    *,
    case: Any,
    comparison_mesh: Any,
    metric_dtype: str | None,
    output: Any,
) -> tuple[Any, str] | tuple[None, None]:
    if comparison_mesh is not None and metric_dtype is not None:
        return comparison_mesh, metric_dtype
    if case is not None and output is not None:
        return build_comparison_mesh(case, predictions, output)
    return None, None


def _scalar_drag_lift_error(ground_truth: dict, predictions: dict, key: str) -> float:
    g = ground_truth.get(key)
    p = predictions.get(key)
    if g is None or p is None:
        return float("nan")
    gt = float(g)
    pr = float(p)
    denom = abs(gt)
    if denom < 1e-14:
        return abs(pr - gt)
    return abs(pr - gt) / denom


def drag_error(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    coeff: float = 1.0,
    drag_direction: list[float] | None = None,
    **_: object,
) -> float:
    mesh, dtype = _resolve_mesh(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    dd = drag_direction if drag_direction is not None else [1.0, 0.0, 0.0]
    if mesh is None or output is None:
        return _scalar_drag_lift_error(ground_truth, predictions, "drag")
    gtp = output.ground_truth_mesh_field_names["pressure"]
    gtw = output.ground_truth_mesh_field_names["shear_stress"]
    prp = output.mesh_field_names["pressure"]
    prw = output.mesh_field_names["shear_stress"]
    try:
        cd_gt, *_ = compute_drag_and_lift(
            mesh,
            pressure_field=gtp,
            wss_field=gtw,
            coeff=coeff,
            drag_direction=dd,
            dtype=dtype,
        )
        cd_pr, *_ = compute_drag_and_lift(
            mesh,
            pressure_field=prp,
            wss_field=prw,
            coeff=coeff,
            drag_direction=dd,
            dtype=dtype,
        )
        denom = abs(cd_gt)
        if denom < 1e-14:
            return float(abs(cd_pr - cd_gt))
        return float(abs(cd_pr - cd_gt) / denom)
    except Exception:
        return _scalar_drag_lift_error(ground_truth, predictions, "drag")


def lift_error(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    coeff: float = 1.0,
    lift_direction: list[float] | None = None,
    **_: object,
) -> float:
    mesh, dtype = _resolve_mesh(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    ld = lift_direction if lift_direction is not None else [0.0, 0.0, 1.0]
    if mesh is None or output is None:
        return _scalar_drag_lift_error(ground_truth, predictions, "lift")
    gtp = output.ground_truth_mesh_field_names["pressure"]
    gtw = output.ground_truth_mesh_field_names["shear_stress"]
    prp = output.mesh_field_names["pressure"]
    prw = output.mesh_field_names["shear_stress"]
    try:
        _, _, _, cl_gt, _, _ = compute_drag_and_lift(
            mesh,
            pressure_field=gtp,
            wss_field=gtw,
            coeff=coeff,
            drag_direction=[1.0, 0.0, 0.0],
            lift_direction=ld,
            dtype=dtype,
        )
        _, _, _, cl_pr, _, _ = compute_drag_and_lift(
            mesh,
            pressure_field=prp,
            wss_field=prw,
            coeff=coeff,
            drag_direction=[1.0, 0.0, 0.0],
            lift_direction=ld,
            dtype=dtype,
        )
        denom = abs(cl_gt)
        if denom < 1e-14:
            return float(abs(cl_pr - cl_gt))
        return float(abs(cl_pr - cl_gt) / denom)
    except Exception:
        return _scalar_drag_lift_error(ground_truth, predictions, "lift")


def register_force_metrics() -> None:
    register_metric("drag_error", drag_error)
    register_metric("lift_error", lift_error)
