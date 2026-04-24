# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model checkpoint / stats resolution (Hugging Face Hub, S3, local)."""

from physicsnemo.cfd.evaluation.assets.builtin_packages import (
    BENCHMARK_CHECKPOINTS_HF_ROOT,
    BUILTIN_MODEL_PACKAGE_ROOTS,
    DOMINO_PACKAGE_ROOT,
    FIGNET_PACKAGE_ROOT,
    GEOTRANSOLVER_PACKAGE_ROOT,
    TRANSOLVER_PACKAGE_ROOT,
    XMGN_PACKAGE_ROOT,
    register_builtin_model_packages,
)
from physicsnemo.cfd.evaluation.assets.package import Package, maybe_touch_hf_config_json
from physicsnemo.cfd.evaluation.assets.registry import (
    AssetSpec,
    clear_default_assets_for_testing,
    get_default_asset,
    register_default_asset,
)
from physicsnemo.cfd.evaluation.assets.resolve import resolve_model_assets

register_builtin_model_packages()

__all__ = [
    "AssetSpec",
    "BENCHMARK_CHECKPOINTS_HF_ROOT",
    "BUILTIN_MODEL_PACKAGE_ROOTS",
    "DOMINO_PACKAGE_ROOT",
    "FIGNET_PACKAGE_ROOT",
    "GEOTRANSOLVER_PACKAGE_ROOT",
    "Package",
    "TRANSOLVER_PACKAGE_ROOT",
    "XMGN_PACKAGE_ROOT",
    "clear_default_assets_for_testing",
    "get_default_asset",
    "maybe_touch_hf_config_json",
    "register_builtin_model_packages",
    "register_default_asset",
    "resolve_model_assets",
]
