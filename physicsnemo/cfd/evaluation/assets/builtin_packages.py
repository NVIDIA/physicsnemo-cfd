# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default remote package roots for built-in benchmark model names.

Weights are resolved when ``model.checkpoint`` and ``model.stats_path`` are omitted and the
wrapper sets ``REQUIRES_REMOTE_ASSETS`` (see :func:`~physicsnemo.cfd.evaluation.assets.resolve.resolve_model_assets`).

**NGC:** ``ngc://`` is not implemented in :class:`~physicsnemo.cfd.evaluation.assets.package.Package`.
Mirror artifacts to Hugging Face (or local paths) and set the per-model ``*_PACKAGE_ROOT`` constants below.

**Naming:** Surface Hub bundles use ``*_surface`` (e.g. ``geotransolver_surface``, ``xmgn_surface``);
volume bundles use ``*_volume``. GeoTransolver / Transolver / DoMINO ship separate checkpoint trees on HF.

Companion files use ``{checkpoint_parent}/…`` in ``extra_resolve_relpaths`` where needed (see
:mod:`~physicsnemo.cfd.evaluation.assets.registry`).
"""

from __future__ import annotations

from physicsnemo.cfd.evaluation.assets.registry import AssetSpec, register_default_asset

# Per-model package roots (``hf://``, ``s3://``, ``file://``, or absolute directory).
GEOTRANSOLVER_PACKAGE_ROOT = "hf://nvidia/geotransolver_drivaerml@main"
TRANSOLVER_PACKAGE_ROOT = "hf://nvidia/transolver_drivaerml@main"
XMGN_PACKAGE_ROOT = "hf://nvidia/xmgn_drivaerml_surface@main"
FIGNET_PACKAGE_ROOT = "hf://nvidia/figconvnet_drivaerml_surface@main"
DOMINO_PACKAGE_ROOT = "hf://nvidia/domino_drivaerml@main"

# Backward-compatible alias (GeoTransolver root).
BENCHMARK_CHECKPOINTS_HF_ROOT = GEOTRANSOLVER_PACKAGE_ROOT

BUILTIN_MODEL_PACKAGE_ROOTS: dict[str, str] = {
    "geotransolver_surface": GEOTRANSOLVER_PACKAGE_ROOT,
    "geotransolver_volume": GEOTRANSOLVER_PACKAGE_ROOT,
    "transolver_surface": TRANSOLVER_PACKAGE_ROOT,
    "transolver_volume": TRANSOLVER_PACKAGE_ROOT,
    "xmgn_surface": XMGN_PACKAGE_ROOT,
    "fignet_surface": FIGNET_PACKAGE_ROOT,
    "domino_surface": DOMINO_PACKAGE_ROOT,
    "domino_volume": DOMINO_PACKAGE_ROOT,
}


def register_builtin_model_packages() -> None:
    """Register :class:`AssetSpec` defaults for matrix benchmark models (idempotent)."""

    register_default_asset(
        "geotransolver_surface",
        AssetSpec(
            package_root=GEOTRANSOLVER_PACKAGE_ROOT,
            checkpoint_relpath="geotransolver_drivaerml_surface_checkpoint/GeoTransolver.0.501.mdlus",
            stats_relpath="geotransolver_drivaerml_surface_checkpoint/global_stats.json",
        ),
    )
    register_default_asset(
        "geotransolver_volume",
        AssetSpec(
            package_root=GEOTRANSOLVER_PACKAGE_ROOT,
            checkpoint_relpath="geotransolver_drivaerml_volume_checkpoint/GeoTransolver.0.501.mdlus",
            stats_relpath="geotransolver_drivaerml_volume_checkpoint/global_stats.json",
        ),
    )

    register_default_asset(
        "transolver_surface",
        AssetSpec(
            package_root=TRANSOLVER_PACKAGE_ROOT,
            checkpoint_relpath="transolver_drivaerml_surface_checkpoint/Transolver.0.501.mdlus",
            stats_relpath="transolver_drivaerml_surface_checkpoint/global_stats.json",
        ),
    )
    register_default_asset(
        "transolver_volume",
        AssetSpec(
            package_root=TRANSOLVER_PACKAGE_ROOT,
            checkpoint_relpath="transolver_drivaerml_volume_checkpoint/Transolver.0.501.mdlus",
            stats_relpath="transolver_drivaerml_volume_checkpoint/global_stats.json",
        ),
    )

    register_default_asset(
        "xmgn_surface",
        AssetSpec(
            package_root=XMGN_PACKAGE_ROOT,
            checkpoint_relpath="final_model_checkpoint.pth",
            stats_relpath="global_stats.json",
        ),
    )
    register_default_asset(
        "fignet_surface",
        AssetSpec(
            package_root=FIGNET_PACKAGE_ROOT,
            checkpoint_relpath="model_00999.pth",
            stats_relpath="global_stats.json",
        ),
    )

    register_default_asset(
        "domino_surface",
        AssetSpec(
            package_root=DOMINO_PACKAGE_ROOT,
            checkpoint_relpath="domino_drivaerml_surface_checkpoint/DoMINO.0.501.mdlus",
            stats_relpath="domino_drivaerml_surface_checkpoint/global_stats.json",
            extra_resolve_relpaths=(
                ("domino_config", "{checkpoint_parent}/config.yaml"),
                ("_resolved_scaling_factors", "{checkpoint_parent}/scaling_factors.pkl"),
            ),
        ),
    )
    register_default_asset(
        "domino_volume",
        AssetSpec(
            package_root=DOMINO_PACKAGE_ROOT,
            checkpoint_relpath="domino_drivaerml_volume_checkpoint/DoMINO.0.501.mdlus",
            stats_relpath="domino_drivaerml_volume_checkpoint/global_stats.json",
            extra_resolve_relpaths=(
                ("domino_config", "{checkpoint_parent}/config.yaml"),
                ("_resolved_scaling_factors", "{checkpoint_parent}/scaling_factors.pkl"),
            ),
        ),
    )
