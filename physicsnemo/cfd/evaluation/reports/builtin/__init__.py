# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register built-in visuals."""

from physicsnemo.cfd.evaluation.reports.builtin.surface_volume import (
    register_plot_fields_volume,
    register_field_comparison_surface,
)


def register_all_builtin_visuals() -> None:
    register_field_comparison_surface()
    register_plot_fields_volume()


register_all_builtin_visuals()
