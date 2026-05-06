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

"""Register built-in visuals."""

from physicsnemo.cfd.evaluation.reports.builtin.design_plots import (
    register_design_visuals,
)
from physicsnemo.cfd.evaluation.reports.builtin.hexbin import (
    register_projections_hexbin,
)
from physicsnemo.cfd.evaluation.reports.builtin.line_plot import register_line_plot
from physicsnemo.cfd.evaluation.reports.builtin.streamlines_visual import (
    register_streamlines_visual,
)
from physicsnemo.cfd.evaluation.reports.builtin.surface_volume import (
    register_plot_fields_volume,
    register_field_comparison_surface,
)
from physicsnemo.cfd.evaluation.reports.builtin.aggregate_volume import (
    register_aggregate_volume,
)


def register_all_builtin_visuals() -> None:
    """Register every built-in visual (surface/volume comparisons, line plots, hexbin, streamlines, aggregate)."""
    register_field_comparison_surface()
    register_plot_fields_volume()
    register_line_plot()
    register_design_visuals()
    register_projections_hexbin()
    register_streamlines_visual()
    register_aggregate_volume()


register_all_builtin_visuals()
