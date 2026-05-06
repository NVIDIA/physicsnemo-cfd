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

"""Streamline comparison: ``streamlines_comparison`` supports volume (point) and surface (cell) meshes."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from physicsnemo.cfd.postprocessing_tools.metrics.streamlines import compute_streamlines
from physicsnemo.cfd.postprocessing_tools.visualization.utils import plot_streamlines
from physicsnemo.cfd.evaluation.config import Config, OutputConfig
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.reports.context_helpers import (
    get_comparison_mesh_for_case,
)
from physicsnemo.cfd.evaluation.reports.registry import register_visual
from physicsnemo.cfd.evaluation.reports.visual_filenames import benchmark_visual_png


def streamlines_comparison(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    context: dict[str, Any] | None = None,
    case_ids: list[str] | None = None,
    canonical_key: str | None = None,
    view: str = "xy",
    **kwargs: Any,
) -> None:
    """GT vs pred streamlines on comparison meshes.

    Dispatches on each case's ``metric_dtype``:

    - ``point`` — volume mesh; uses ``ground_truth_volume_mesh_field_names`` /
      ``volume_mesh_field_names`` (typically ``velocity``).
    - ``cell`` — surface PolyData; uses ``ground_truth_mesh_field_names`` / ``mesh_field_names``
      (e.g. ``shear_stress`` for wall-shear / skin-friction lines).

    ``canonical_key`` defaults to ``output.streamlines_vector_canonical`` (usually ``velocity``).
    For surface runs, set ``canonical_key`` in YAML (e.g. ``shear_stress``).
    """
    out = Path(output_dir) / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    output: OutputConfig = config.output
    ck = canonical_key or output.streamlines_vector_canonical

    for run_idx, run in enumerate(results):
        if run.get("skipped"):
            continue
        model = run["model"]
        dataset = run["dataset"]
        for row in run.get("per_case") or []:
            cid = row["case_id"]
            if case_ids is not None and cid not in case_ids:
                continue
            md = row.get("metric_dtype")
            if md == "point":
                if (
                    ck not in output.ground_truth_volume_mesh_field_names
                    or ck not in output.volume_mesh_field_names
                ):
                    log_dataset(
                        "benchmark",
                        f"Skip streamlines_comparison for {cid!r}: canonical_key {ck!r} missing from "
                        f"volume field maps (ground_truth_volume_mesh_field_names / volume_mesh_field_names)",
                    )
                    continue
                gt_name = output.ground_truth_volume_mesh_field_names[ck]
                pred_name = output.volume_mesh_field_names[ck]
                fname_tag = "streamlines"
            elif md == "cell":
                if (
                    ck not in output.ground_truth_mesh_field_names
                    or ck not in output.mesh_field_names
                ):
                    log_dataset(
                        "benchmark",
                        f"Skip streamlines_comparison for {cid!r}: canonical_key {ck!r} missing from "
                        f"surface field maps (ground_truth_mesh_field_names / mesh_field_names)",
                    )
                    continue
                gt_name = output.ground_truth_mesh_field_names[ck]
                pred_name = output.mesh_field_names[ck]
                fname_tag = "streamlines_surface"
            else:
                log_dataset(
                    "benchmark",
                    f"Skip streamlines_comparison for {cid!r}: unsupported metric_dtype {md!r}",
                )
                continue

            mesh = get_comparison_mesh_for_case(row, cid, run_idx, context)
            if mesh is None:
                log_dataset(
                    "benchmark",
                    f"Skip streamlines_comparison for {cid!r}: no comparison mesh",
                )
                continue
            m1 = mesh.copy(deep=True)
            m2 = mesh.copy(deep=True)
            # Volume metrics use point dof (md == "point"); integrate in 3-D. Surface (md == "cell"):
            # PyVista surface-constrained streamline mode for boundary vector fields (e.g. wall shear).
            surface_sl = md == "cell"
            try:
                sl_true = compute_streamlines(
                    m1, gt_name, surface_streamlines=surface_sl
                )
                sl_pred = compute_streamlines(
                    m2, pred_name, surface_streamlines=surface_sl
                )
            except Exception:
                log_dataset(
                    "benchmark",
                    f"compute_streamlines failed for case {cid!r} ({model=} {dataset=}):\n"
                    f"{traceback.format_exc()}",
                )
                continue
            plotter = plot_streamlines(
                sl_true, sl_pred, geometry=mesh, view=view, **kwargs
            )
            safe = benchmark_visual_png(model, dataset, cid, fname_tag)
            plotter.screenshot(str(out / safe))
            plotter.close()
            log_dataset("benchmark", f"Wrote streamlines {out / safe}")


def register_streamlines_visual() -> None:
    """Register the ``streamlines_comparison`` visual (and back-compat alias)."""
    register_visual("streamlines_comparison", streamlines_comparison)
    # Backward-compatible alias (same implementation; surface uses cell dtype + surface field maps).
    register_visual("streamlines_comparison_surface", streamlines_comparison)
