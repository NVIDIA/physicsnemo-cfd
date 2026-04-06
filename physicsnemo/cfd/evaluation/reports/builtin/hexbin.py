# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hexbin projection visual (``plot_projections_hexbin``) from explicit mesh paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pyvista as pv

from physicsnemo.cfd.postprocessing_tools.visualization.utils import plot_projections_hexbin
from physicsnemo.cfd.evaluation.config import Config
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.reports.registry import register_visual


def projections_hexbin(
    config: Config,
    results: list[dict[str, Any]],
    output_dir: str,
    *,
    context: dict[str, Any] | None = None,
    mesh_paths: list[str] | None = None,
    field: str = "p_error",
    direction: str = "XY",
    grid_size: int = 50,
    **_kwargs: Any,
) -> None:
    """Aggregate hexbin over multiple meshes (paths on disk). *results* / *context* unused."""
    del context, results, config
    if not mesh_paths:
        raise ValueError("projections_hexbin requires non-empty ``mesh_paths`` (list of VTK paths).")
    meshes = [pv.read(p) for p in mesh_paths]
    fig = plot_projections_hexbin(meshes, field, direction, grid_size=grid_size)
    out = Path(output_dir) / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    safe = f"hexbin_{field}_{direction}.png".replace("/", "_")
    out_png = out / safe
    fig.savefig(str(out_png), bbox_inches="tight", dpi=300)
    plt.close(fig)
    log_dataset("benchmark", f"Wrote hexbin {out_png}")


def register_projections_hexbin() -> None:
    register_visual("projections_hexbin", projections_hexbin)
