# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default remote package roots for built-in benchmark model names.

Weights are resolved when ``model.checkpoint`` and ``model.stats_path`` are omitted and the
wrapper sets ``REQUIRES_REMOTE_ASSETS`` (see :func:`~physicsnemo.cfd.evaluation.assets.resolve.resolve_model_assets`).

**NGC:** ``ngc://`` is not implemented in :class:`~physicsnemo.cfd.evaluation.assets.package.Package`.
Mirror artifacts to Hugging Face (or local paths) and set the per-model ``*_PACKAGE_ROOT`` constants below.

**Layout:** Each per-model HF repo has its own structure. GeoTransolver, Transolver, and DoMINO
nest checkpoints under ``<model>_drivaerml_surface_checkpoint/``; XMGN and FIGConvNet store files
flat at the repo root. If a model's weights move, adjust ``checkpoint_relpath`` / ``stats_relpath``
in :func:`register_builtin_model_packages`.
"""

from __future__ import annotations

from physicsnemo.cfd.evaluation.assets.registry import AssetSpec, register_default_asset

# Per-model package roots (``hf://``, ``s3://``, ``file://``, or absolute directory).
GEOTRANSOLVER_PACKAGE_ROOT = "hf://nvidia/geotransolver_drivaerml@main"
TRANSOLVER_PACKAGE_ROOT = "hf://nvidia/transolver_drivaerml@main"
XMGN_PACKAGE_ROOT = "hf://nvidia/xmgn_drivaerml_surface@main"
FIGNET_PACKAGE_ROOT = "hf://nvidia/figconvnet_drivaerml_surface@main"
DOMINO_PACKAGE_ROOT = "hf://nvidia/domino_drivaerml@main"

# Backward-compatible alias.
BENCHMARK_CHECKPOINTS_HF_ROOT = GEOTRANSOLVER_PACKAGE_ROOT

BUILTIN_MODEL_PACKAGE_ROOTS: dict[str, str] = {
    "geotransolver": GEOTRANSOLVER_PACKAGE_ROOT,
    "transolver": TRANSOLVER_PACKAGE_ROOT,
    "xmgn": XMGN_PACKAGE_ROOT,
    "fignet": FIGNET_PACKAGE_ROOT,
    "domino": DOMINO_PACKAGE_ROOT,
}


def register_builtin_model_packages() -> None:
    """Register :class:`AssetSpec` defaults for matrix benchmark models (idempotent)."""

    register_default_asset(
        "geotransolver",
        AssetSpec(
            package_root=GEOTRANSOLVER_PACKAGE_ROOT,
            checkpoint_relpath="geotransolver_drivaerml_surface_checkpoint/checkpoint.0.501.pt",
            stats_relpath="geotransolver_drivaerml_surface_checkpoint/global_stats.json",
        ),
    )
    register_default_asset(
        "transolver",
        AssetSpec(
            package_root=TRANSOLVER_PACKAGE_ROOT,
            checkpoint_relpath="transolver_drivaerml_surface_checkpoint/checkpoint.0.501.pt",
            stats_relpath="transolver_drivaerml_surface_checkpoint/global_stats.json",
        ),
    )
    register_default_asset(
        "xmgn",
        AssetSpec(
            package_root=XMGN_PACKAGE_ROOT,
            checkpoint_relpath="final_model_checkpoint.pth",
            stats_relpath="global_stats.json",
        ),
    )
    register_default_asset(
        "fignet",
        AssetSpec(
            package_root=FIGNET_PACKAGE_ROOT,
            checkpoint_relpath="model_00999.pth",
            stats_relpath="global_stats.json",
        ),
    )
    register_default_asset(
        "domino",
        AssetSpec(
            package_root=DOMINO_PACKAGE_ROOT,
            checkpoint_relpath="domino_drivaerml_surface_checkpoint/checkpoint.0.501.pt",
            stats_relpath="domino_drivaerml_surface_checkpoint/global_stats.json",
            extra_resolve_relpaths=(
                (
                    "domino_config",
                    "domino_drivaerml_surface_checkpoint/config.yaml",
                ),
            ),
        ),
    )
