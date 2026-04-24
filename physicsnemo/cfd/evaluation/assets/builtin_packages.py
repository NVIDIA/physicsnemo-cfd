# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default remote package roots for built-in benchmark model names.

Weights are resolved when ``model.checkpoint`` and ``model.stats_path`` are omitted and the
wrapper sets ``REQUIRES_REMOTE_ASSETS`` (see :func:`~physicsnemo.cfd.evaluation.assets.resolve.resolve_model_assets`).

**NGC:** ``ngc://`` is not implemented in :class:`~physicsnemo.cfd.evaluation.assets.package.Package`.
Mirror artifacts to Hugging Face (or local paths) and set the per-model ``*_PACKAGE_ROOT`` constants below.

**Layout:** Each ``package_root`` is expected to contain the checkpoint tree for that model, e.g.::

    benchmark_models/geotransolver_drivaerml_surface_checkpoint/checkpoint.0.501.pt
    benchmark_models/geotransolver_drivaerml_surface_checkpoint/global_stats.json

If a model’s weights live in a different repo or path layout, change only that model’s
:data:`*_PACKAGE_ROOT` (and adjust ``checkpoint_relpath`` / ``stats_relpath`` in
:func:`register_builtin_model_packages` if needed).
"""

from __future__ import annotations

from physicsnemo.cfd.evaluation.assets.registry import AssetSpec, register_default_asset

# Default Hub root when every model shares one repository (override per model as needed).
_DEFAULT_SHARED_BENCHMARK_HF_ROOT = (
    "hf://nvidia/physicsnemo-cfd-external-aero-benchmark-checkpoints@main"
)

# Per-model package roots (``hf://``, ``s3://``, ``file://``, or absolute directory).
GEOTRANSOLVER_PACKAGE_ROOT = _DEFAULT_SHARED_BENCHMARK_HF_ROOT
TRANSOLVER_PACKAGE_ROOT = _DEFAULT_SHARED_BENCHMARK_HF_ROOT
XMGN_PACKAGE_ROOT = _DEFAULT_SHARED_BENCHMARK_HF_ROOT
FIGNET_PACKAGE_ROOT = _DEFAULT_SHARED_BENCHMARK_HF_ROOT
DOMINO_PACKAGE_ROOT = _DEFAULT_SHARED_BENCHMARK_HF_ROOT

# Backward-compatible alias: same as :data:`GEOTRANSOLVER_PACKAGE_ROOT` by default.
BENCHMARK_CHECKPOINTS_HF_ROOT = GEOTRANSOLVER_PACKAGE_ROOT

BUILTIN_MODEL_PACKAGE_ROOTS: dict[str, str] = {
    "geotransolver": GEOTRANSOLVER_PACKAGE_ROOT,
    "transolver": TRANSOLVER_PACKAGE_ROOT,
    "xmgn": XMGN_PACKAGE_ROOT,
    "fignet": FIGNET_PACKAGE_ROOT,
    "domino": DOMINO_PACKAGE_ROOT,
}


def _bm(*parts: str) -> str:
    return "/".join(("benchmark_models",) + parts)


def register_builtin_model_packages() -> None:
    """Register :class:`AssetSpec` defaults for matrix benchmark models (idempotent)."""

    register_default_asset(
        "geotransolver",
        AssetSpec(
            package_root=GEOTRANSOLVER_PACKAGE_ROOT,
            checkpoint_relpath=_bm(
                "geotransolver_drivaerml_surface_checkpoint",
                "checkpoint.0.501.pt",
            ),
            stats_relpath=_bm(
                "geotransolver_drivaerml_surface_checkpoint",
                "global_stats.json",
            ),
        ),
    )
    register_default_asset(
        "transolver",
        AssetSpec(
            package_root=TRANSOLVER_PACKAGE_ROOT,
            checkpoint_relpath=_bm(
                "transolver_drivaerml_surface_checkpoint",
                "checkpoint.0.501.pt",
            ),
            stats_relpath=_bm(
                "transolver_drivaerml_surface_checkpoint",
                "global_stats.json",
            ),
        ),
    )
    register_default_asset(
        "xmgn",
        AssetSpec(
            package_root=XMGN_PACKAGE_ROOT,
            checkpoint_relpath=_bm(
                "xmgn_drivaerml_surface_checkpoint",
                "final_model_checkpoint.pth",
            ),
            stats_relpath=_bm(
                "xmgn_drivaerml_surface_checkpoint",
                "global_stats.json",
            ),
        ),
    )
    register_default_asset(
        "fignet",
        AssetSpec(
            package_root=FIGNET_PACKAGE_ROOT,
            checkpoint_relpath=_bm(
                "fignet_drivaerml_surface_checkpoint",
                "model_00999.pth",
            ),
            stats_relpath=_bm(
                "fignet_drivaerml_surface_checkpoint",
                "global_stats.json",
            ),
        ),
    )
    register_default_asset(
        "domino",
        AssetSpec(
            package_root=DOMINO_PACKAGE_ROOT,
            checkpoint_relpath=_bm(
                "domino_drivaerml_surface_checkpoint",
                "checkpoint.0.501.pt",
            ),
            stats_relpath=_bm(
                "domino_drivaerml_surface_checkpoint",
                "global_stats.json",
            ),
            extra_resolve_relpaths=(
                (
                    "domino_config",
                    _bm("domino_drivaerml_surface_checkpoint", "config.yaml"),
                ),
            ),
        ),
    )
