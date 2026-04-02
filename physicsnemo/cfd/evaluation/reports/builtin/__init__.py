# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register built-in visuals."""

from physicsnemo.cfd.evaluation.reports.builtin.design_plots import register_design_visuals
from physicsnemo.cfd.evaluation.reports.builtin.hexbin import register_projections_hexbin
from physicsnemo.cfd.evaluation.reports.builtin.line_plot import register_line_plot
from physicsnemo.cfd.evaluation.reports.builtin.streamlines_visual import register_streamlines_visual
from physicsnemo.cfd.evaluation.reports.builtin.surface_volume import (
    register_plot_fields_volume,
    register_field_comparison_surface,
)


def register_all_builtin_visuals() -> None:
    register_field_comparison_surface()
    register_plot_fields_volume()
    register_line_plot()
    register_design_visuals()
    register_projections_hexbin()
    register_streamlines_visual()


register_all_builtin_visuals()
