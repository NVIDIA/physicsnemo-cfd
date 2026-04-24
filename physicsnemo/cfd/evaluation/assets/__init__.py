# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model checkpoint / stats resolution (Hugging Face Hub, S3, local)."""

from physicsnemo.cfd.evaluation.assets.package import Package, maybe_touch_hf_config_json
from physicsnemo.cfd.evaluation.assets.registry import (
    AssetSpec,
    clear_default_assets_for_testing,
    get_default_asset,
    register_default_asset,
)
from physicsnemo.cfd.evaluation.assets.resolve import resolve_model_assets

__all__ = [
    "AssetSpec",
    "Package",
    "clear_default_assets_for_testing",
    "get_default_asset",
    "maybe_touch_hf_config_json",
    "register_default_asset",
    "resolve_model_assets",
]
