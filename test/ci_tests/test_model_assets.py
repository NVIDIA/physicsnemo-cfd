# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evaluation.assets (local Package, resolve_model_assets, cache fingerprint)."""

from __future__ import annotations

from pathlib import Path

import pytest

from physicsnemo.cfd.evaluation.assets import (
    AssetSpec,
    BENCHMARK_CHECKPOINTS_HF_ROOT,
    BUILTIN_MODEL_PACKAGE_ROOTS,
    Package,
    TRANSOLVER_PACKAGE_ROOT,
    clear_default_assets_for_testing,
    get_default_asset,
    register_builtin_model_packages,
    register_default_asset,
    resolve_model_assets,
)
from physicsnemo.cfd.evaluation.benchmarks.metrics_cache import metrics_cache_fingerprint
from physicsnemo.cfd.evaluation.config import ModelConfig
from physicsnemo.cfd.evaluation.models.wrappers.surface_baseline import SurfaceBaselineWrapper


def test_package_resolve_local_file(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    root.mkdir()
    f = root / "global_stats.json"
    f.write_text("{}")
    pkg = Package(str(root))
    assert Path(pkg.resolve("global_stats.json")) == f.resolve()


def test_package_resolve_local_missing(tmp_path: Path) -> None:
    pkg = Package(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        pkg.resolve("missing.pt")


class _TrainedStub:
    REQUIRES_REMOTE_ASSETS = True


def test_builtin_geotransolver_surface_asset_registered() -> None:
    spec = get_default_asset("geotransolver_surface")
    assert spec is not None
    assert spec.package_root == BENCHMARK_CHECKPOINTS_HF_ROOT
    assert spec.package_root == BUILTIN_MODEL_PACKAGE_ROOTS["geotransolver_surface"]
    assert spec.checkpoint_relpath.endswith(".mdlus")
    assert "surface_checkpoint" in spec.checkpoint_relpath
    assert spec.stats_relpath.endswith("global_stats.json")


def test_builtin_transolver_uses_per_model_package_root() -> None:
    spec = get_default_asset("transolver_surface")
    assert spec is not None
    assert spec.package_root == TRANSOLVER_PACKAGE_ROOT
    assert spec.package_root == BUILTIN_MODEL_PACKAGE_ROOTS["transolver_surface"]


def test_builtin_xmgn_fignet_surface_assets_registered() -> None:
    xspec = get_default_asset("xmgn_surface")
    assert xspec is not None
    assert xspec.package_root == BUILTIN_MODEL_PACKAGE_ROOTS["xmgn_surface"]
    assert xspec.checkpoint_relpath.endswith(".pth")

    fspec = get_default_asset("fignet_surface")
    assert fspec is not None
    assert fspec.package_root == BUILTIN_MODEL_PACKAGE_ROOTS["fignet_surface"]


def test_builtin_geotransolver_volume_asset_registered() -> None:
    spec = get_default_asset("geotransolver_volume")
    assert spec is not None
    assert spec.package_root == BUILTIN_MODEL_PACKAGE_ROOTS["geotransolver_volume"]
    assert spec.checkpoint_relpath.endswith(".mdlus")
    assert "volume_checkpoint" in spec.checkpoint_relpath
    assert spec.stats_relpath.endswith("global_stats.json")


def test_builtin_domino_extra_assets_use_checkpoint_parent_placeholder() -> None:
    """Built-in DoMINO registers companion YAML/pickle paths via ``{checkpoint_parent}`` templates."""
    spec = get_default_asset("domino_surface")
    assert spec is not None
    assert len(spec.extra_resolve_relpaths) == 2
    assert any(
        "{checkpoint_parent}" in rel for _, rel in spec.extra_resolve_relpaths
    )
    kinds = {k for k, _ in spec.extra_resolve_relpaths}
    assert "domino_config" in kinds and "_resolved_scaling_factors" in kinds


def test_resolve_model_assets_explicit_paths() -> None:
    cfg = ModelConfig(
        name="fignet_surface",
        checkpoint="/abs/ck.pt",
        stats_path="/abs/stats.json",
    )
    ck, st, aid, xkw = resolve_model_assets(cfg, _TrainedStub)
    assert ck == "/abs/ck.pt" and st == "/abs/stats.json" and aid is None and xkw == {}


def test_resolve_model_assets_missing_raises() -> None:
    """Model name with no builtin or explicit paths must raise."""
    cfg = ModelConfig(name="_not_a_builtin_benchmark_model", checkpoint="", stats_path="")
    with pytest.raises(ValueError, match="checkpoint"):
        resolve_model_assets(cfg, _TrainedStub)


def test_resolve_model_assets_baseline_no_paths() -> None:
    cfg = ModelConfig(name="surface_baseline", checkpoint="", stats_path="")
    ck, st, aid, xkw = resolve_model_assets(cfg, SurfaceBaselineWrapper)
    assert ck == "" and st == "" and aid is None and xkw == {}


def test_resolve_model_assets_via_registry(tmp_path: Path) -> None:
    clear_default_assets_for_testing()
    root = tmp_path / "hub_sim"
    root.mkdir()
    (root / "w.pt").write_text("x")
    (root / "stats.json").write_text("{}")
    try:
        register_default_asset(
            "_ci_test_model",
            AssetSpec(
                package_root=str(root),
                checkpoint_relpath="w.pt",
                stats_relpath="stats.json",
                extra_resolve_relpaths=(("domino_config", "cfg.yaml"),),
            ),
        )
        (root / "cfg.yaml").write_text("model:\n  model_type: surface\n")
        cfg = ModelConfig(name="_ci_test_model", checkpoint="", stats_path="")
        ck, st, aid, xkw = resolve_model_assets(cfg, _TrainedStub)
        assert aid == f"package:{root}|w.pt|stats.json|domino_config:cfg.yaml"
        assert Path(ck).name == "w.pt"
        assert Path(st).name == "stats.json"
        assert Path(xkw["domino_config"]).name == "cfg.yaml"
    finally:
        clear_default_assets_for_testing()
        register_builtin_model_packages()


def test_metrics_cache_fingerprint_asset_identity_changes_digest() -> None:
    base = dict(
        model_name="m",
        model_checkpoint="",
        model_stats_path="",
        model_kwargs={},
        model_inference_domain="surface",
        model_asset_identity="package:hf://a/b@main|ck|st",
        dataset_name="d",
        dataset_root="/r",
        dataset_split=None,
        dataset_kwargs_resolved={},
        output_dict={},
        metric_specs=[("l2_pressure", {})],
    )
    fp1 = metrics_cache_fingerprint(**base)
    fp2 = metrics_cache_fingerprint(**{**base, "model_asset_identity": "package:hf://a/b@main|ck2|st"})
    assert fp1 != fp2

    fp_explicit = metrics_cache_fingerprint(
        **{**base, "model_asset_identity": None, "model_checkpoint": "a.pt", "model_stats_path": "b.json"},
    )
    assert fp_explicit != fp1
