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

"""Force coefficient metrics via physicsnemo.cfd.postprocessing_tools.metrics.aero_forces.

Each metric returns a dict that expands in the benchmark engine to:

- ``drag_error`` — relative |Cd_pred − Cd_true| / |Cd_true| (or absolute if |Cd_true| ≈ 0)
- ``drag_true`` / ``drag_pred`` — integrated **drag coefficient Cd** (GT vs pred fields)
- ``lift_error`` — relative |Cl_pred − Cl_true| / |Cl_true|
- ``lift_true`` / ``lift_pred`` — integrated **lift coefficient Cl**

Use the ``*_true`` / ``*_pred`` keys with ``design_scatter`` / ``design_trend`` in evaluation config.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from physicsnemo.cfd.postprocessing_tools.metric_registry import register_metric
from physicsnemo.cfd.postprocessing_tools.metrics.aero_forces import (
    compute_drag_and_lift,
)
from physicsnemo.cfd.evaluation.metrics.mesh_bridge import (
    resolve_comparison_mesh_for_metric,
)
from physicsnemo.cfd.evaluation.metrics.metric_exceptions import (
    RECOVERABLE_MESH_METRIC_ERRORS,
)

_LOG = logging.getLogger(__name__)


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
    """Relative error (``"error"`` sub-key) plus Cd or Cl pair (``true`` / ``pred`` sub-keys)."""
    denom = abs(cd_gt)
    if denom < 1e-14:
        rel = float(abs(cd_pr - cd_gt))
    else:
        rel = float(abs(cd_pr - cd_gt) / denom)
    return {"error": rel, "true": float(cd_gt), "pred": float(cd_pr)}


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
    """Drag coefficient relative error plus integrated ``Cd`` (true / pred) on the comparison mesh."""
    mesh, dtype = resolve_comparison_mesh_for_metric(
        predictions,
        case=case,
        comparison_mesh=comparison_mesh,
        metric_dtype=metric_dtype,
        output=output,
    )
    dd = drag_direction if drag_direction is not None else [1.0, 0.0, 0.0]
    if mesh is None or output is None:
        rel, gt, pr = _scalar_drag_lift_triplet(ground_truth, predictions, "drag")
        return {"error": rel, "true": gt, "pred": pr}
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
    except RECOVERABLE_MESH_METRIC_ERRORS:
        _LOG.warning(
            "compute_drag_and_lift failed for drag_error; using scalar drag/lift metadata if present:\n%s",
            traceback.format_exc(),
        )
        rel, gt, pr = _scalar_drag_lift_triplet(ground_truth, predictions, "drag")
        return {"error": rel, "true": gt, "pred": pr}


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
    """Lift coefficient relative error plus integrated ``Cl`` (true / pred) on the comparison mesh."""
    mesh, dtype = resolve_comparison_mesh_for_metric(
        predictions,
        case=case,
        comparison_mesh=comparison_mesh,
        metric_dtype=metric_dtype,
        output=output,
    )
    ld = lift_direction if lift_direction is not None else [0.0, 0.0, 1.0]
    if mesh is None or output is None:
        rel, gt, pr = _scalar_drag_lift_triplet(ground_truth, predictions, "lift")
        return {"error": rel, "true": gt, "pred": pr}
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
    except RECOVERABLE_MESH_METRIC_ERRORS:
        _LOG.warning(
            "compute_drag_and_lift failed for lift_error; using scalar drag/lift metadata if present:\n%s",
            traceback.format_exc(),
        )
        rel, gt, pr = _scalar_drag_lift_triplet(ground_truth, predictions, "lift")
        return {"error": rel, "true": gt, "pred": pr}


def register_force_metrics() -> None:
    """Register the built-in surface ``drag`` / ``lift`` metrics."""
    register_metric("drag", drag_error, domain="surface")
    register_metric("lift", lift_error, domain="surface")
