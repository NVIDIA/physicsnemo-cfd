# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2-style metrics delegating to physicsnemo.cfd.bench.metrics.l2_errors."""

from __future__ import annotations

from typing import Any

import numpy as np

from physicsnemo.cfd.bench.metric_registry import register_metric
from physicsnemo.cfd.bench.metrics.l2_errors import (
    compute_area_weighted_l2_errors,
    compute_l2_errors,
)
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


def _l2_pressure_numpy(
    ground_truth: dict,
    predictions: dict,
    mask: np.ndarray | None = None,
    **_: object,
) -> float:
    gt = np.asarray(ground_truth.get("pressure", []), dtype=np.float64).flatten()
    pred = np.asarray(predictions.get("pressure", []), dtype=np.float64).flatten()
    if gt.size == 0 or pred.size == 0 or gt.shape != pred.shape:
        return float("nan")
    if mask is not None:
        m = np.asarray(mask).flatten()
        gt = gt[m]
        pred = pred[m]
    denom = np.linalg.norm(gt)
    if denom < 1e-14:
        return float(np.linalg.norm(pred - gt))
    return float(np.linalg.norm(pred - gt) / denom)


def _l2_shear_numpy(
    ground_truth: dict,
    predictions: dict,
    mask: np.ndarray | None = None,
    **_: object,
) -> float:
    gt = np.asarray(ground_truth.get("shear_stress", []), dtype=np.float64)
    pred = np.asarray(predictions.get("shear_stress", []), dtype=np.float64)
    if gt.size == 0 or pred.size == 0 or gt.shape != pred.shape:
        return float("nan")
    gt = gt.flatten()
    pred = pred.flatten()
    if mask is not None:
        m = np.asarray(mask).flatten()
        if m.dtype == bool and m.size * 3 == gt.size:
            gt = gt.reshape(-1, 3)[m].flatten()
            pred = pred.reshape(-1, 3)[m].flatten()
        else:
            gt = gt[m]
            pred = pred[m]
    denom = np.linalg.norm(gt)
    if denom < 1e-14:
        return float(np.linalg.norm(pred - gt))
    return float(np.linalg.norm(pred - gt) / denom)


def l2_pressure(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    mask: np.ndarray | None = None,
    **_: object,
) -> float:
    mesh, dtype = _resolve_mesh(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    if mesh is None or output is None:
        return _l2_pressure_numpy(ground_truth, predictions, mask=mask)
    gtn = output.ground_truth_mesh_field_names["pressure"]
    prn = output.mesh_field_names["pressure"]
    try:
        d = compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
        key = f"{gtn}_l2_error"
        return float(d[key]) if key in d else float("nan")
    except Exception:
        return float("nan")


def l2_shear_stress(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    mask: np.ndarray | None = None,
    **_: object,
) -> dict[str, float]:
    mesh, dtype = _resolve_mesh(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    if mesh is None or output is None:
        v = _l2_shear_numpy(ground_truth, predictions, mask=mask)
        return {"magnitude": v}
    gtn = output.ground_truth_mesh_field_names["shear_stress"]
    prn = output.mesh_field_names["shear_stress"]
    try:
        return compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
    except Exception:
        return {"magnitude": float("nan")}


def l2_pressure_area_weighted(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    **_: object,
) -> float:
    mesh, dtype = _resolve_mesh(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    if mesh is None or output is None:
        return float("nan")
    gtn = output.ground_truth_mesh_field_names["pressure"]
    prn = output.mesh_field_names["pressure"]
    try:
        d = compute_area_weighted_l2_errors(mesh, [gtn], [prn], dtype=dtype)
        key = f"{gtn}_area_wt_l2_error"
        return float(d[key]) if key in d else float("nan")
    except Exception:
        return float("nan")


def l2_pressure_volume(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    gt_key: str = "pressure_volume",
    pred_key: str | None = None,
    mask: np.ndarray | None = None,
    **_: object,
) -> float:
    mesh, dtype = _resolve_mesh(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    pk = pred_key if pred_key is not None else gt_key
    if mesh is None or output is None:
        gt = np.asarray(ground_truth.get(gt_key, []), dtype=np.float64).flatten()
        pred = np.asarray(predictions.get(pk, []), dtype=np.float64).flatten()
        if gt.size == 0 or pred.size == 0 or gt.shape != pred.shape:
            return float("nan")
        if mask is not None:
            m = np.asarray(mask).flatten()
            gt = gt[m]
            pred = pred[m]
        denom = np.linalg.norm(gt)
        if denom < 1e-14:
            return float(np.linalg.norm(pred - gt))
        return float(np.linalg.norm(pred - gt) / denom)
    gtn = output.ground_truth_volume_mesh_field_names[gt_key]
    prn = output.volume_mesh_field_names[pk]
    try:
        d = compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
        key = f"{gtn}_l2_error"
        return float(d[key]) if key in d else float("nan")
    except Exception:
        return float("nan")


def l2_velocity(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    **_: object,
) -> dict[str, float]:
    mesh, dtype = _resolve_mesh(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    if mesh is None or output is None:
        gt = np.asarray(ground_truth.get("velocity", []), dtype=np.float64)
        pred = np.asarray(predictions.get("velocity", []), dtype=np.float64)
        if gt.size == 0 or pred.size == 0 or gt.shape != pred.shape:
            return {"magnitude": float("nan")}
        gt = gt.flatten()
        pred = pred.flatten()
        denom = np.linalg.norm(gt)
        if denom < 1e-14:
            mag = float(np.linalg.norm(pred - gt))
        else:
            mag = float(np.linalg.norm(pred - gt) / denom)
        return {"magnitude": mag}
    gtn = output.ground_truth_volume_mesh_field_names["velocity"]
    prn = output.volume_mesh_field_names["velocity"]
    try:
        return compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
    except Exception:
        return {"magnitude": float("nan")}


def l2_turbulent_viscosity(
    ground_truth: dict,
    predictions: dict,
    *,
    case: Any = None,
    comparison_mesh: Any = None,
    metric_dtype: str | None = None,
    output: Any = None,
    gt_key: str = "turbulent_viscosity",
    pred_key: str | None = None,
    mask: np.ndarray | None = None,
    **_: object,
) -> float:
    mesh, dtype = _resolve_mesh(
        predictions, case=case, comparison_mesh=comparison_mesh, metric_dtype=metric_dtype, output=output
    )
    pk = pred_key if pred_key is not None else gt_key
    if mesh is None or output is None:
        gt = np.asarray(ground_truth.get(gt_key, []), dtype=np.float64).flatten()
        pred = np.asarray(predictions.get(pk, []), dtype=np.float64).flatten()
        if gt.size == 0 or pred.size == 0 or gt.shape != pred.shape:
            return float("nan")
        if mask is not None:
            m = np.asarray(mask).flatten()
            gt = gt[m]
            pred = pred[m]
        denom = np.linalg.norm(gt)
        if denom < 1e-14:
            return float(np.linalg.norm(pred - gt))
        return float(np.linalg.norm(pred - gt) / denom)
    gtn = output.ground_truth_volume_mesh_field_names[gt_key]
    prn = output.volume_mesh_field_names[pk]
    try:
        d = compute_l2_errors(mesh, [gtn], [prn], dtype=dtype)
        key = f"{gtn}_l2_error"
        return float(d[key]) if key in d else float("nan")
    except Exception:
        return float("nan")


def register_l2_metrics() -> None:
    for name, fn in [
        ("l2_pressure", l2_pressure),
        ("l2_shear_stress", l2_shear_stress),
        ("l2_pressure_area_weighted", l2_pressure_area_weighted),
        ("area_wt_l2_pressure", l2_pressure_area_weighted),
        ("l2_pressure_volume", l2_pressure_volume),
        ("l2_velocity", l2_velocity),
        ("l2_turbulent_viscosity", l2_turbulent_viscosity),
    ]:
        register_metric(name, fn)
