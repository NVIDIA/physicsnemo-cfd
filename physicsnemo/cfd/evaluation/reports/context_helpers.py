# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
