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
