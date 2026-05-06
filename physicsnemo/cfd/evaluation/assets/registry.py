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

"""Built-in default model asset locations (Hugging Face / future NGC). Third parties may register."""

from __future__ import annotations

from dataclasses import dataclass

#: model_name -> spec when ``checkpoint`` / ``stats_path`` are omitted in config.
_DEFAULT_ASSETS: dict[str, "AssetSpec"] = {}


@dataclass(frozen=True)
class AssetSpec:
    """Paths relative to :attr:`package_root` passed to :class:`~physicsnemo.cfd.evaluation.assets.package.Package`.

    Parameters
    ----------
    extra_resolve_relpaths
        Optional ``(load_kwarg_name, relpath_in_package)`` pairs resolved with the same
        :class:`~physicsnemo.cfd.evaluation.assets.package.Package` as checkpoint/stats.

        A relpath may contain the literal substring ``{checkpoint_parent}``; it is replaced with the
        parent directory of ``checkpoint_relpath`` (after resolving overrides), so companion assets
        can live next to the weight file without duplicating folder names in registrations.
    """

    package_root: str
    checkpoint_relpath: str
    stats_relpath: str
    extra_resolve_relpaths: tuple[tuple[str, str], ...] = ()


def register_default_asset(model_name: str, spec: AssetSpec) -> None:
    """Register (or replace) the default HF/local package for a model name."""
    _DEFAULT_ASSETS[model_name] = spec


def get_default_asset(model_name: str) -> AssetSpec | None:
    """Return the registered default asset for ``model_name`` (or ``None`` if unknown)."""
    return _DEFAULT_ASSETS.get(model_name)


def clear_default_assets_for_testing() -> None:
    """Remove all registered defaults (tests only)."""
    _DEFAULT_ASSETS.clear()
