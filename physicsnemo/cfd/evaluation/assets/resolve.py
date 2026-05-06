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

"""Resolve ``ModelConfig`` checkpoint/stats paths and optional companion assets via :class:`Package`.

Companion relpaths in ``AssetSpec.extra_resolve_relpaths`` support ``{checkpoint_parent}``; see
:class:`~physicsnemo.cfd.evaluation.assets.registry.AssetSpec`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Type

from physicsnemo.cfd.evaluation.assets.package import (
    Package,
    maybe_touch_hf_config_json,
)
from physicsnemo.cfd.evaluation.assets.registry import AssetSpec, get_default_asset
from physicsnemo.cfd.evaluation.config import ModelConfig

_ASSET_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _sanitize_cache_token(name: str) -> str:
    s = _ASSET_SAFE.sub("_", name.strip())
    return s[:64] if s else "model"


_CHECKPOINT_PARENT = "{checkpoint_parent}"


def _expand_checkpoint_parent(rel_template: str, ck_rel: str) -> str:
    """Substitute ``{checkpoint_parent}`` with the directory containing the checkpoint file."""
    if _CHECKPOINT_PARENT not in rel_template:
        return rel_template
    parent = Path(ck_rel).parent.as_posix()
    if not parent or parent == ".":
        raise ValueError(
            f"checkpoint_relpath {ck_rel!r} must include a directory segment when using "
            f"{_CHECKPOINT_PARENT!r} in ``AssetSpec.extra_resolve_relpaths``."
        )
    return rel_template.replace(_CHECKPOINT_PARENT, parent)


def _asset_identity(
    pkg_root: str, ck_rel: str, st_rel: str, spec: AssetSpec | None
) -> str:
    ident = f"package:{pkg_root}|{ck_rel}|{st_rel}"
    if spec and spec.extra_resolve_relpaths:
        for k, rel_t in spec.extra_resolve_relpaths:
            rel_exp = _expand_checkpoint_parent(rel_t, ck_rel)
            ident += f"|{k}:{rel_exp}"
    return ident


def resolve_model_assets(
    model_config: ModelConfig,
    wrapper_class: Type[Any],
) -> tuple[str, str, str | None, dict[str, str]]:
    """
    Return local ``checkpoint_path``, ``stats_path``, optional ``asset_identity`` for metrics cache,
    and ``load_kw`` (companion assets merged before ``wrapper.load``).

    When ``asset_identity`` is not ``None``, callers should pass it to :func:`metrics_cache_fingerprint`
    and may pass empty checkpoint/stats strings for the fingerprint payload to avoid cache-dir churn.

    ``load_kw`` is merged as ``{**load_kw, **model.merged_kwargs_for_load()}`` so YAML kwargs override
    package-resolved paths.

    Raises
    ------
    ValueError
        If trained weights are required but paths and package cannot be resolved.
    """
    requires = getattr(wrapper_class, "REQUIRES_REMOTE_ASSETS", True)
    if not requires:
        return model_config.checkpoint, model_config.stats_path, None, {}

    ck = (model_config.checkpoint or "").strip()
    st = (model_config.stats_path or "").strip()

    spec = get_default_asset(model_config.name)

    pkg_root = (model_config.package or "").strip()
    if not pkg_root:
        pkg_root = str(model_config.kwargs.get("package") or "").strip()
    if not pkg_root and spec is not None:
        pkg_root = spec.package_root.strip()

    # Explicit user paths (both required for trained models without package)
    if ck and st:
        return ck, st, None, {}

    if not pkg_root:
        raise ValueError(
            f"Model {model_config.name!r}: set both ``model.checkpoint`` and ``model.stats_path``, "
            f"or set ``model.package`` (e.g. hf://org/repo@revision), or call "
            f"``register_default_asset`` for this model name."
        )

    yaml_ck = (model_config.checkpoint_relpath or "").strip() or str(
        model_config.kwargs.get("checkpoint_relpath") or ""
    ).strip()
    yaml_st = (model_config.stats_relpath or "").strip() or str(
        model_config.kwargs.get("stats_relpath") or ""
    ).strip()
    if bool(yaml_ck) ^ bool(yaml_st):
        raise ValueError(
            f"Model {model_config.name!r}: set both ``checkpoint_relpath`` and ``stats_relpath``, "
            f"or omit both to use registered defaults."
        )

    ck_rel = yaml_ck or ((spec.checkpoint_relpath or "").strip() if spec else "")
    st_rel = yaml_st or ((spec.stats_relpath or "").strip() if spec else "")
    if not ck_rel or not st_rel:
        raise ValueError(
            f"Model {model_config.name!r}: package {pkg_root!r} requires "
            f"``checkpoint_relpath`` and ``stats_relpath`` (on ``model`` or in ``model.kwargs``, "
            f"or via ``AssetSpec`` for defaults)."
        )

    cache_sub = _sanitize_cache_token(model_config.name)
    pkg = Package(
        pkg_root,
        cache_options={
            "cache_storage": Package.default_cache(f"packages/{cache_sub}"),
        },
    )
    maybe_touch_hf_config_json(pkg)
    ck_path = pkg.resolve(ck_rel)
    st_path = pkg.resolve(st_rel)
    load_kw: dict[str, str] = {}
    if spec and spec.extra_resolve_relpaths:
        for kw_name, rel_t in spec.extra_resolve_relpaths:
            rel_exp = _expand_checkpoint_parent(rel_t, ck_rel)
            load_kw[kw_name] = pkg.resolve(rel_exp)
    identity = _asset_identity(pkg_root, ck_rel, st_rel, spec)
    return ck_path, st_path, identity, load_kw
