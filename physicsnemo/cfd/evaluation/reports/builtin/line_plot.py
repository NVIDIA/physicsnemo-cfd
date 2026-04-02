# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Line plot visual wrapping ``physicsnemo.cfd.bench.visualization.utils.plot_line``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pyvista as pv

from physicsnemo.cfd.bench.visualization.utils import plot_line
from physicsnemo.cfd.evaluation.config import Config, OutputConfig
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.reports.registry import register_visual


def _resolve_gt_pred_fields(output: OutputConfig, canonical_key: str) -> tuple[str, str]:
    """Surface or volume VTK names for a canonical key."""
    if canonical_key in output.ground_truth_mesh_field_names and canonical_key in output.mesh_field_names:
        return (
            output.ground_truth_mesh_field_names[canonical_key],
            output.mesh_field_names[canonical_key],
        )
    if canonical_key in output.ground_truth_volume_mesh_field_names and canonical_key in output.volume_mesh_field_names:
        return (
            output.ground_truth_volume_mesh_field_names[canonical_key],
            output.volume_mesh_field_names[canonical_key],
        )
    raise KeyError(
        f"Canonical key {canonical_key!r} not found in surface or volume output field maps"
    )


def _comparison_mesh_to_line_polydata(mesh: pv.DataSet) -> pv.PolyData:
    """One point per cell with cell fields on points (for ``plot_line`` single-line branch)."""
    if mesh.n_cells > 0:
        return mesh.cell_centers(vertex=False)
    return mesh


def line_plot(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    case_ids: list[str] | None = None,
    canonical_key: str = "pressure",
    plot_coord: str = "x",
    normalize_factor: float = 1.0,
    coord_trim: tuple[float | None, float | None] | None = None,
    field_trim: tuple[float | None, float | None] | None = None,
    flip: bool = False,
    **kwargs: Any,
) -> None:
    """GT vs pred line plot along ``plot_coord`` using cell-centered samples of the comparison mesh.

    Resolves VTK array names from ``output.ground_truth_mesh_field_names`` and
    ``output.mesh_field_names`` for ``canonical_key`` (surface defaults). The mesh is reduced to
    cell centers (or points if there are no cells), then passed to ``plot_line`` as a **single**
    polyline-like dataset (values vs sorted ``plot_coord``).

    Extra ``**kwargs`` are forwarded to ``plot_line`` (e.g. ``true_line_kwargs``, ``pred_line_kwargs``,
    ``xlabel``, ``ylabel``, ``title_kwargs``).
    """
    out = Path(output_dir)
    vis_dir = out / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    output: OutputConfig = config.output

    field_true, field_pred = _resolve_gt_pred_fields(output, canonical_key)

    ct_raw = coord_trim if coord_trim is not None else (None, None)
    ft_raw = field_trim if field_trim is not None else (None, None)
    ct = (ct_raw[0], ct_raw[1]) if isinstance(ct_raw, (list, tuple)) and len(ct_raw) == 2 else (None, None)
    ft = (ft_raw[0], ft_raw[1]) if isinstance(ft_raw, (list, tuple)) and len(ft_raw) == 2 else (None, None)

    for run in results:
        if run.get("skipped"):
            continue
        model = run["model"]
        dataset = run["dataset"]
        for row in run.get("per_case") or []:
            cid = row["case_id"]
            if case_ids is not None and cid not in case_ids:
                continue
            path = row.get("comparison_mesh_path")
            if not path:
                log_dataset(
                    "benchmark",
                    f"Skip line_plot for {cid!r}: no comparison_mesh_path",
                )
                continue
            mesh = pv.read(path)
            line_mesh = _comparison_mesh_to_line_polydata(mesh)
            fig = plot_line(
                line_mesh,
                plot_coord=plot_coord,
                field_true=field_true,
                field_pred=field_pred,
                normalize_factor=normalize_factor,
                coord_trim=ct,
                field_trim=ft,
                flip=flip,
                **kwargs,
            )
            safe = f"{model}_{dataset}_{cid}_line_{canonical_key}_{plot_coord}.png".replace(
                "/", "_"
            )
            out_png = vis_dir / safe
            fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
            plt.close(fig)
            log_dataset("benchmark", f"Wrote line plot {out_png}")


def register_line_plot() -> None:
    register_visual("line_plot", line_plot)
    register_visual("plot_line", line_plot)  # alias (same implementation)
