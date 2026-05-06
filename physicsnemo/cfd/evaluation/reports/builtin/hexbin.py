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

"""Hexbin projection visual (``plot_projections_hexbin``) from explicit mesh paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pyvista as pv

from physicsnemo.cfd.postprocessing_tools.visualization.utils import (
    plot_projections_hexbin,
)
from physicsnemo.cfd.evaluation.config import Config
from physicsnemo.cfd.evaluation.datasets.progress import log_dataset
from physicsnemo.cfd.evaluation.reports.registry import register_visual
from physicsnemo.cfd.evaluation.reports.visual_filenames import benchmark_visual_png


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
    coordinate_layout: str = "world",
    **_kwargs: Any,
) -> None:
    """Aggregate hexbin over multiple meshes (paths on disk). *results* / *context* unused.

    ``coordinate_layout`` selects how mesh vertex columns map to the plane (see
    ``plot_projections_hexbin``): ``world`` for Cartesian (x,y,z) comparison meshes,
    ``embedding`` when columns 0-1 hold pre-projected axes (legacy VTK).
    """
    del context, results, config
    if not mesh_paths:
        raise ValueError(
            "projections_hexbin requires non-empty ``mesh_paths`` (list of VTK paths)."
        )
    meshes = [pv.read(p) for p in mesh_paths]
    fig = plot_projections_hexbin(
        meshes,
        field,
        direction,
        grid_size=grid_size,
        coordinate_layout=coordinate_layout,
    )
    out = Path(output_dir) / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    safe = benchmark_visual_png("hexbin", field, direction)
    out_png = out / safe
    fig.savefig(str(out_png), bbox_inches="tight", dpi=300)
    plt.close(fig)
    log_dataset("benchmark", f"Wrote hexbin {out_png}")


def register_projections_hexbin() -> None:
    """Register the ``projections_hexbin`` visual."""
    register_visual("projections_hexbin", projections_hexbin)
