# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Force coefficient metrics via physicsnemo.cfd.bench.metrics.aero_forces.

Each metric returns a dict that expands in the benchmark engine to:

- ``drag_error`` — relative |Cd_pred − Cd_true| / |Cd_true| (or absolute if |Cd_true| ≈ 0)
- ``drag_error_true`` / ``drag_error_pred`` — integrated **drag coefficient Cd** (GT vs pred fields)
- ``lift_error`` — relative |Cl_pred − Cl_true| / |Cl_true|
- ``lift_error_true`` / ``lift_error_pred`` — integrated **lift coefficient Cl**

Use the ``*_true`` / ``*_pred`` keys with ``design_scatter`` / ``design_trend`` in evaluation config.
"""

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
    rel, _, _ = _scalar_drag_lift_triplet(ground_truth, predictions, key)
    return rel


def _scalar_drag_lift_triplet(
    ground_truth: dict, predictions: dict, key: str
) -> tuple[float, float, float]:
    """Relative error and (true, pred) scalars for ``key`` (e.g. drag/lift from case metadata)."""
    g = ground_truth.get(key)
    p = predictions.get(key)
    if g is None or p is None:
        return float("nan"), float("nan"), float("nan")
    gt = float(g)
    pr = float(p)
    denom = abs(gt)
    if denom < 1e-14:
        rel = abs(pr - gt)
    else:
        rel = abs(pr - gt) / denom
    return rel, gt, pr


def _rel_and_pair(cd_gt: float, cd_pr: float) -> dict[str, float]:
    """Relative error (``""`` → scalar metric name) plus Cd or Cl pair (``true`` / ``pred`` sub-keys)."""
    denom = abs(cd_gt)
    if denom < 1e-14:
        rel = float(abs(cd_pr - cd_gt))
    else:
        rel = float(abs(cd_pr - cd_gt) / denom)
    return {"": rel, "true": float(cd_gt), "pred": float(cd_pr)}


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
) -> dict[str, float]:
    mesh, dtype = _resolve_mesh(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    dd = drag_direction if drag_direction is not None else [1.0, 0.0, 0.0]
    if mesh is None or output is None:
        rel, gt, pr = _scalar_drag_lift_triplet(ground_truth, predictions, "drag")
        return {"": rel, "true": gt, "pred": pr}
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
        return _rel_and_pair(float(cd_gt), float(cd_pr))
    except Exception:
        rel, gt, pr = _scalar_drag_lift_triplet(ground_truth, predictions, "drag")
        return {"": rel, "true": gt, "pred": pr}


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
) -> dict[str, float]:
    mesh, dtype = _resolve_mesh(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    ld = lift_direction if lift_direction is not None else [0.0, 0.0, 1.0]
    if mesh is None or output is None:
        rel, gt, pr = _scalar_drag_lift_triplet(ground_truth, predictions, "lift")
        return {"": rel, "true": gt, "pred": pr}
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
        return _rel_and_pair(float(cl_gt), float(cl_pr))
    except Exception:
        rel, gt, pr = _scalar_drag_lift_triplet(ground_truth, predictions, "lift")
        return {"": rel, "true": gt, "pred": pr}


def register_force_metrics() -> None:
    register_metric("drag_error", drag_error)
    register_metric("lift_error", lift_error)
