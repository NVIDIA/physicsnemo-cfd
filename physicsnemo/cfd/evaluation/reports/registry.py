# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named visual/report plugin registry (metrics-style) for post-scalar benchmark outputs."""

from __future__ import annotations

from typing import Any, Callable

# Visual plugins: (config, results, output_dir, *, context=None, **per_visual_kwargs) -> None.
# ``context`` may include ``comparison_meshes_by_run``: list[dict[case_id, pyvista.DataSet]]
# aligned with ``results`` (see ``run_optional_report_plugins``).
VisualFn = Callable[..., None]

_REGISTRY: dict[str, VisualFn] = {}


def register_visual(name: str, fn: VisualFn) -> None:
    """Register a visual plugin by name."""
    _REGISTRY[name] = fn


def get_visual(name: str) -> VisualFn:
    """Resolve a visual function by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown visual: {name}. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_visuals() -> list[str]:
    """Return registered visual names."""
    return sorted(_REGISTRY.keys())


def normalize_visuals_config(
    visuals: list[str] | list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """Return list of (visual_name, kwargs) from YAML (same rules as metrics)."""
    out: list[tuple[str, dict[str, Any]]] = []
    for v in visuals:
        if isinstance(v, str):
            out.append((v, {}))
        elif isinstance(v, dict) and "name" in v:
            name = v["name"]
            kwargs = {k: val for k, val in v.items() if k != "name"}
            out.append((name, kwargs))
        else:
            raise ValueError(f"Invalid visual entry: {v}")
    return out
