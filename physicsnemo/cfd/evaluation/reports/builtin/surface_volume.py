# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in visuals wrapping ``physicsnemo.cfd.postprocessing_tools.visualization.utils``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyvista as pv

from physicsnemo.cfd.postprocessing_tools.visualization.utils import plot_field_comparisons, plot_fields
from physicsnemo.cfd.evaluation.config import Config, OutputConfig
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.reports.context_helpers import get_comparison_mesh_for_case
from physicsnemo.cfd.evaluation.reports.registry import register_visual


def field_comparison_surface(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    context: dict[str, Any] | None = None,
    case_ids: list[str] | None = None,
    canonical_keys: list[str] | None = None,
    view: str = "xy",
    **kwargs: Any,
) -> None:
    """Plot GT vs pred field comparison for surface comparison meshes (pressure default)."""
    out = Path(output_dir)
    vis_dir = out / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    keys = canonical_keys or ["pressure"]
    output: OutputConfig = config.output

    for run_idx, run in enumerate(results):
        if run.get("skipped"):
            continue
        model = run["model"]
        dataset = run["dataset"]
        for row in run.get("per_case") or []:
            cid = row["case_id"]
            if case_ids is not None and cid not in case_ids:
                continue
            dtype = row.get("metric_dtype") or "cell"
            mesh = get_comparison_mesh_for_case(row, cid, run_idx, context)
            if mesh is None:
                log_dataset(
                    "benchmark",
                    f"Skip field_comparison_surface for {cid!r}: no comparison mesh in context or path",
                )
                continue
            true_fields: list[str] = []
            pred_fields: list[str] = []
            for k in keys:
                if k not in output.ground_truth_mesh_field_names:
                    raise KeyError(f"output.ground_truth_mesh_field_names missing {k!r}")
                if k not in output.mesh_field_names:
                    raise KeyError(f"output.mesh_field_names missing {k!r}")
                true_fields.append(output.ground_truth_mesh_field_names[k])
                pred_fields.append(output.mesh_field_names[k])
            plotter = plot_field_comparisons(
                mesh,
                true_fields,
                pred_fields,
                view=view,
                dtype=dtype,
                **kwargs,
            )
            safe = f"{model}_{dataset}_{cid}_field_comparison.png".replace("/", "_")
            plotter.screenshot(str(vis_dir / safe))
            plotter.close()


def plot_fields_volume(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    context: dict[str, Any] | None = None,
    case_ids: list[str] | None = None,
    fields: list[str] | None = None,
    view: str = "xy",
    **kwargs: Any,
) -> None:
    """Plot selected VTK fields on volume comparison meshes (``plot_fields``)."""
    out = Path(output_dir)
    vis_dir = out / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    output: OutputConfig = config.output
    field_list = fields
    if field_list is None:
        field_list = list(output.volume_mesh_field_names.values())

    for run_idx, run in enumerate(results):
        if run.get("skipped"):
            continue
        model = run["model"]
        dataset = run["dataset"]
        for row in run.get("per_case") or []:
            cid = row["case_id"]
            if case_ids is not None and cid not in case_ids:
                continue
            dtype = row.get("metric_dtype") or "point"
            mesh = get_comparison_mesh_for_case(row, cid, run_idx, context)
            if mesh is None:
                log_dataset(
                    "benchmark",
                    f"Skip plot_fields_volume for {cid!r}: no comparison mesh in context or path",
                )
                continue
            plotter = plot_fields(
                mesh,
                field_list,
                view=view,
                dtype=dtype,
                **kwargs,
            )
            safe = f"{model}_{dataset}_{cid}_plot_fields.png".replace("/", "_")
            plotter.screenshot(str(vis_dir / safe))
            plotter.close()


def register_field_comparison_surface() -> None:
    register_visual("field_comparison_surface", field_comparison_surface)


def register_plot_fields_volume() -> None:
    register_visual("plot_fields_volume", plot_fields_volume)
