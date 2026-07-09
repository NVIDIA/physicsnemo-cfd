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

"""Tests for the MC weight-perturbation proxy's ``perturb_scope`` parser.

Requires torch (imported by the wrapper module); skipped in the CPU-only lint env, runs on the
GPU node. This only exercises the pure parameter-selection logic — no model / forward pass.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from physicsnemo.cfd.evaluation.models.wrappers.mc_perturbation.wrapper import (  # noqa: E402
    _parse_perturb_scope,
)

# A representative GeoTransolver-like parameter-name layout: 20 transformer blocks + heads.
_NAMES = (
    ["preprocess.weight", "preprocess.bias"]
    + [f"blocks.{i}.attn.qkv.weight" for i in range(20)]
    + [f"blocks.{i}.mlp.fc1.weight" for i in range(20)]
    + ["head.weight", "head.bias"]
)


def test_scope_all_selects_everything() -> None:
    assert _parse_perturb_scope("all", _NAMES) == set(_NAMES)
    assert _parse_perturb_scope("", _NAMES) == set(_NAMES)


def test_scope_last_n_layers_selects_top_block_indices() -> None:
    sel = _parse_perturb_scope("last_n_layers=2", _NAMES)
    # Blocks 18 and 19 (the two highest indices) across both attn and mlp param groups.
    assert sel == {
        "blocks.18.attn.qkv.weight",
        "blocks.19.attn.qkv.weight",
        "blocks.18.mlp.fc1.weight",
        "blocks.19.mlp.fc1.weight",
    }


def test_scope_fraction_selects_last_tensors() -> None:
    sel = _parse_perturb_scope("fraction=0.5", _NAMES)
    assert len(sel) == round(0.5 * len(_NAMES))
    # Fraction takes the *tail* of the tensor list (includes the head params).
    assert "head.weight" in sel and "preprocess.weight" not in sel


def test_scope_last_n_layers_falls_back_when_no_indices() -> None:
    names = ["a.weight", "b.weight", "c.weight", "d.weight"]
    sel = _parse_perturb_scope("last_n_layers=2", names)
    # No numeric block indices -> fall back to last 25% of tensors (>=1).
    assert sel == {"d.weight"}


def test_scope_bad_value_defaults_gracefully() -> None:
    sel = _parse_perturb_scope("fraction=notanumber", _NAMES)
    assert len(sel) == round(0.25 * len(_NAMES))
