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

"""Resolve comparison meshes for report visuals: in-memory context first, then disk."""

from __future__ import annotations

from typing import Any

import pyvista as pv


def get_comparison_mesh_for_case(
    row: dict[str, Any],
    case_id: str,
    run_idx: int,
    context: dict[str, Any] | None,
) -> pv.DataSet | None:
    """Return comparison mesh from ``context['comparison_meshes_by_run']`` or ``comparison_mesh_path``."""
    ctx = context or {}
    by_run = ctx.get("comparison_meshes_by_run")
    if isinstance(by_run, list) and run_idx < len(by_run):
        m = by_run[run_idx].get(case_id)
        if m is not None:
            return m
    path = row.get("comparison_mesh_path")
    if path:
        return pv.read(path)
    return None
