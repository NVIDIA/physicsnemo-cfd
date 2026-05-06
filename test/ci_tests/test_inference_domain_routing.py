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

"""Routing of ``surface`` vs ``volume`` matches ``load()`` when ``model.inference_domain`` is omitted."""

from __future__ import annotations

import pytest

from physicsnemo.cfd.evaluation.benchmarks.engine import _effective_inference_domain
from physicsnemo.cfd.evaluation.config import ModelConfig
from physicsnemo.cfd.evaluation.datasets.adapters.drivaerml import DrivAerMLAdapter
from physicsnemo.cfd.evaluation.models import get_model_wrapper
from physicsnemo.cfd.evaluation.models.model_registry import (
    get_inference_domain_for_model,
)


def test_get_inference_domain_for_model_rejects_wrong_class_inference_domain() -> None:
    """Invalid ``INFERENCE_DOMAIN`` on the wrapper class raises; no silent default."""
    cls = get_model_wrapper("transolver_surface")
    old = cls.INFERENCE_DOMAIN
    try:
        cls.INFERENCE_DOMAIN = "volum"  # type: ignore[misc, assignment]
        with pytest.raises(ValueError) as exc:
            get_inference_domain_for_model("transolver_surface")
        assert "INFERENCE_DOMAIN" in str(exc.value)
    finally:
        cls.INFERENCE_DOMAIN = old


def test_get_inference_domain_for_model_dual_mode_without_static_attr_raises() -> None:
    """Dual-mode wrappers (``INFERENCE_DOMAIN is None``) raise rather than silently defaulting."""
    with pytest.raises(ValueError, match="INFERENCE_DOMAIN is None"):
        get_inference_domain_for_model("transolver_surface")


def test_effective_inference_domain_transolver_omitted_matches_load_default_surface() -> (
    None
):
    """No YAML inference_domain → same ``surface`` default as ``TransolverWrapper.load``."""
    mc = ModelConfig(name="transolver_surface", kwargs={})
    assert _effective_inference_domain(mc) == "surface"


def test_effective_inference_domain_explicit_model_field() -> None:
    """An explicit ``model.inference_domain`` field wins over wrapper defaults."""
    mc = ModelConfig(name="transolver_surface", inference_domain="volume")
    assert _effective_inference_domain(mc) == "volume"


def test_effective_inference_domain_from_merged_kwargs(tmp_path) -> None:
    """``model.kwargs.inference_domain`` when top-level field unset."""
    mc = ModelConfig(name="transolver_surface", kwargs={"inference_domain": "volume"})
    assert _effective_inference_domain(mc) == "volume"


def test_effective_inference_domain_domino_reads_model_type_from_yaml(tmp_path) -> None:
    """DoMINO routes ``model_type`` from ``domino_config`` YAML when the field is omitted."""
    cfg = tmp_path / "domino.yaml"
    cfg.write_text(
        "model:\n  model_type: volume\n",
        encoding="utf-8",
    )
    mc = ModelConfig(name="domino_volume", kwargs={"domino_config": str(cfg)})
    assert _effective_inference_domain(mc) == "volume"


def test_effective_inference_domain_domino_explicit_overrides_yaml(tmp_path) -> None:
    """An explicit ``model.inference_domain`` overrides the DoMINO YAML ``model_type``."""
    cfg = tmp_path / "domino.yaml"
    cfg.write_text(
        "model:\n  model_type: volume\n",
        encoding="utf-8",
    )
    mc = ModelConfig(
        name="domino_volume",
        inference_domain="surface",
        kwargs={"domino_config": str(cfg)},
    )
    assert _effective_inference_domain(mc) == "surface"


def test_effective_inference_domain_normalizes_whitespace() -> None:
    """Inference-domain strings are case-folded and whitespace-stripped."""
    mc = ModelConfig(
        name="transolver_surface", kwargs={"inference_domain": "  Volume "}
    )
    assert _effective_inference_domain(mc) == "volume"


def test_effective_inference_domain_typo_raises() -> None:
    """Typo in ``model.kwargs.inference_domain`` raises ``ValueError`` rather than silently defaulting."""
    with pytest.raises(ValueError):
        _effective_inference_domain(
            ModelConfig(name="transolver_surface", kwargs={"inference_domain": "volum"})
        )


def test_explicit_model_inference_domain_typo_raises() -> None:
    """Typo in the top-level ``model.inference_domain`` field raises ``ValueError``."""
    with pytest.raises(ValueError):
        _effective_inference_domain(
            ModelConfig(name="transolver_surface", inference_domain="volum"),
        )


def test_drivaerml_inference_domain_typo_raises() -> None:
    """Typo in DrivAerML ``dataset.kwargs.inference_domain`` raises ``ValueError``."""
    with pytest.raises(ValueError):
        DrivAerMLAdapter.inference_domain_from_kwargs({"inference_domain": "volum"})
