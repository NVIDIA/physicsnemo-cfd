# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report visuals registry and built-in plot hooks (``physicsnemo.cfd.postprocessing_tools.visualization``).

Pass optional ``context`` with ``comparison_meshes_by_run`` into
``physicsnemo.cfd.evaluation.benchmarks.report_plugins.run_optional_report_plugins`` so mesh-based
built-ins can skip ``pv.read(comparison_mesh_path)``.
"""

from physicsnemo.cfd.evaluation.reports.registry import (
    get_visual,
    list_visuals,
    register_visual,
)

import physicsnemo.cfd.evaluation.reports.builtin  # noqa: F401 — register built-ins

__all__ = ["register_visual", "get_visual", "list_visuals"]
