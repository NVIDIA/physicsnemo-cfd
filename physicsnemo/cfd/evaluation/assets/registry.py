# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in default model asset locations (Hugging Face / future NGC). Third parties may register."""

from __future__ import annotations

from dataclasses import dataclass

#: model_name -> spec when ``checkpoint`` / ``stats_path`` are omitted in config.
_DEFAULT_ASSETS: dict[str, "AssetSpec"] = {}


@dataclass(frozen=True)
class AssetSpec:
    """Paths relative to :attr:`package_root` passed to :class:`~physicsnemo.cfd.evaluation.assets.package.Package`."""

    package_root: str
    checkpoint_relpath: str
    stats_relpath: str


def register_default_asset(model_name: str, spec: AssetSpec) -> None:
    """Register (or replace) the default HF/local package for a model name."""
    _DEFAULT_ASSETS[model_name] = spec


def get_default_asset(model_name: str) -> AssetSpec | None:
    return _DEFAULT_ASSETS.get(model_name)


def clear_default_assets_for_testing() -> None:
    """Remove all registered defaults (tests only)."""
    _DEFAULT_ASSETS.clear()
