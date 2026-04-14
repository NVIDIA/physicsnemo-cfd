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

"""Config schema and loader for inference and benchmarking."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``; overlay wins for non-dict values and new keys."""
    result = dict(base)
    for key, val in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = deep_merge_dict(result[key], val)
        else:
            result[key] = val
    return result


@dataclass
class MetricsCacheConfig:
    """
    Optional on-disk cache of per-case scalar metrics from benchmark runs.

    When enabled, cases with a matching fingerprint skip reading VTK, running
    inference, and recomputing metrics. Meshes and report plots are not cached.

    Attributes
    ----------
    enabled : bool
        Whether to read and write metric cache files.
    path : str
        Cache root directory. If empty while ``enabled`` is True, defaults to
        ``<run.output_dir>/.metrics_cache``.
    """

    enabled: bool = False
    path: str = ""


@dataclass
class RunConfig:
    device: str = "cuda:0"
    output_dir: str = "benchmark_results"
    seed: int = 42
    batch_size: int = 1
    #: If False, inference CLI skips writing ``inference_<model>_<case>.vtp|vtu`` (comparison mesh / visuals unchanged).
    save_inference_mesh: bool = True
    metrics_cache: MetricsCacheConfig = field(default_factory=MetricsCacheConfig)
    #: When True (default) and launched multi-process (e.g. ``torchrun``), shard cases across ranks via
    #: ``DistributedManager`` and merge before reports. When False, each rank runs the full case list (debug only).
    distributed: bool = True
    #: If True, :func:`run_benchmark` raises :class:`BenchmarkPolicyError` when every matrix/single run was skipped.
    fail_on_all_skipped: bool = False
    #: If True, :func:`run_benchmark` raises when any aggregate metric value in ``results[*]["metrics"]`` is NaN.
    fail_on_any_metric_nan: bool = False


@dataclass
class ModelConfig:
    name: str = "fignet"
    checkpoint: str = ""
    stats_path: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    # Optional override; otherwise taken from the registered wrapper's INFERENCE_DOMAIN.
    inference_domain: str | None = None

    def merged_kwargs_for_load(self) -> dict[str, Any]:
        """``model.kwargs`` plus ``inference_domain`` so wrappers can branch surface vs volume."""
        kw = dict(self.kwargs)
        if self.inference_domain in ("surface", "volume"):
            kw["inference_domain"] = self.inference_domain
        return kw


@dataclass
class DatasetConfig:
    name: str = "drivaerml"
    root: str = ""
    split: str | None = None
    case_ids: list[str] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReproducibilityConfig:
    #: When True, write full ``os.environ`` to ``env.json`` under ``run.output_dir`` (avoid in shared CI; may leak secrets).
    log_env: bool = False
    save_artifacts: bool = True


@dataclass
class BenchmarkConfig:
    mode: str = "single"  # "single" | "matrix"
    models: list[ModelConfig] = field(default_factory=list)
    datasets: list[DatasetConfig] = field(default_factory=list)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)


@dataclass
class ReportsConfig:
    """Post-scalar outputs: optional comparison mesh export and registered visuals (plots)."""

    enabled: bool = False
    plugins: list[dict[str, Any]] = field(default_factory=list)
    #: When True, write each case comparison mesh to ``comparison_mesh_subdir`` for downstream visualization.
    save_comparison_meshes: bool = False
    comparison_mesh_subdir: str = "comparison_meshes"
    #: Case IDs for which an in-memory comparison mesh is retained for ``reports.visuals`` (saves RAM on large sweeps).
    #: When ``None``, all cases that build a comparison mesh are kept (legacy behavior). When a non-empty list, only
    #: those IDs are stored in ``comparison_meshes_by_run``. Also used as default ``case_ids`` for any visual that
    #: omits ``case_ids`` (per-visual ``case_ids`` still overrides). Empty list ``[]`` retains no meshes in context
    #: (use ``save_comparison_meshes: true`` if PNGs must load from disk).
    visual_case_ids: list[str] | None = None
    #: Visuals to run (same list style as ``metrics``): strings or ``{name: ..., ...kwargs}``.
    visuals: list[str] | list[dict[str, Any]] = field(default_factory=list)


# Canonical prediction keys used by wrappers; mesh array names are configurable per dataset/convention.
DEFAULT_MESH_FIELD_NAMES: dict[str, str] = {
    "pressure": "pMeanTrimPred",
    "shear_stress": "wallShearStressMeanTrimPred",
}

DEFAULT_VOLUME_MESH_FIELD_NAMES: dict[str, str] = {
    "pressure": "pMeanPred",
    "turbulent_viscosity": "nutMeanPred",
    "velocity": "UMeanPred",
}

DEFAULT_GROUND_TRUTH_MESH_FIELD_NAMES: dict[str, str] = {
    "pressure": "pMean",
    "shear_stress": "wallShearStressMean",
}

DEFAULT_GROUND_TRUTH_VOLUME_MESH_FIELD_NAMES: dict[str, str] = {
    "pressure": "pMean",
    "turbulent_viscosity": "nutMean",
    "velocity": "UMean",
}


@dataclass
class OutputConfig:
    """Output / mesh writing options.

    ``mesh_field_names`` maps canonical surface keys to VTP array names (predictions).
    ``ground_truth_mesh_field_names`` maps canonical keys to reference VTK names on the mesh.
    ``volume_mesh_field_names`` / ``ground_truth_volume_mesh_field_names`` do the same for volume.
    """

    mesh_field_names: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MESH_FIELD_NAMES))
    volume_mesh_field_names: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_VOLUME_MESH_FIELD_NAMES)
    )
    ground_truth_mesh_field_names: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_GROUND_TRUTH_MESH_FIELD_NAMES)
    )
    ground_truth_volume_mesh_field_names: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_GROUND_TRUTH_VOLUME_MESH_FIELD_NAMES)
    )
    #: Canonical key for ``streamlines_comparison`` report visual (volume vector field).
    streamlines_vector_canonical: str = "velocity"


@dataclass
class Config:
    """Root config for inference and benchmarking."""

    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metrics: list[str] | list[dict[str, Any]] = field(default_factory=lambda: ["l2_pressure"])
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    reports: ReportsConfig = field(default_factory=ReportsConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build Config from a nested dict (e.g. from YAML/JSON)."""
        run_raw = dict(data.get("run") or {})
        mc_raw = run_raw.pop("metrics_cache", None)
        if mc_raw is None:
            mc_raw = {}
        elif not isinstance(mc_raw, dict):
            raise TypeError("run.metrics_cache must be a mapping if provided")
        run = RunConfig(
            device=str(run_raw.get("device", "cuda:0")),
            output_dir=str(run_raw.get("output_dir", "benchmark_results")),
            seed=int(run_raw.get("seed", 42)),
            batch_size=int(run_raw.get("batch_size", 1)),
            save_inference_mesh=bool(run_raw.get("save_inference_mesh", True)),
            metrics_cache=MetricsCacheConfig(
                enabled=bool(mc_raw.get("enabled", False)),
                path=str(mc_raw.get("path") or ""),
            ),
            distributed=bool(run_raw.get("distributed", True)),
            fail_on_all_skipped=bool(run_raw.get("fail_on_all_skipped", False)),
            fail_on_any_metric_nan=bool(run_raw.get("fail_on_any_metric_nan", False)),
        )
        model = ModelConfig(**(data.get("model") or {}))
        dataset = DatasetConfig(**(data.get("dataset") or {}))
        out = data.get("output") or {}
        mesh_fn = dict(DEFAULT_MESH_FIELD_NAMES)
        if "mesh_field_names" in out:
            mesh_fn = {**mesh_fn, **out["mesh_field_names"]}
        vol_fn = dict(DEFAULT_VOLUME_MESH_FIELD_NAMES)
        if "volume_mesh_field_names" in out:
            vol_fn = {**vol_fn, **out["volume_mesh_field_names"]}
        gt_mesh = dict(DEFAULT_GROUND_TRUTH_MESH_FIELD_NAMES)
        if "ground_truth_mesh_field_names" in out:
            gt_mesh = {**gt_mesh, **out["ground_truth_mesh_field_names"]}
        gt_vol = dict(DEFAULT_GROUND_TRUTH_VOLUME_MESH_FIELD_NAMES)
        if "ground_truth_volume_mesh_field_names" in out:
            gt_vol = {**gt_vol, **out["ground_truth_volume_mesh_field_names"]}
        output = OutputConfig(
            mesh_field_names=mesh_fn,
            volume_mesh_field_names=vol_fn,
            ground_truth_mesh_field_names=gt_mesh,
            ground_truth_volume_mesh_field_names=gt_vol,
            streamlines_vector_canonical=str(
                out.get("streamlines_vector_canonical") or "velocity"
            ),
        )
        metrics = data.get("metrics", ["l2_pressure"])
        bench = data.get("benchmark") or {}
        rep = bench.get("reproducibility") or {}
        base_model = data.get("model") or {}
        base_dataset = data.get("dataset") or {}
        model_list = []
        for m in bench.get("models", []):
            if isinstance(m, dict):
                model_list.append(ModelConfig(**m))
            else:
                model_list.append(ModelConfig(**{**base_model, "name": str(m)}))
        dataset_list = []
        for d in bench.get("datasets", []):
            if isinstance(d, dict):
                dataset_list.append(DatasetConfig(**d))
            else:
                dataset_list.append(DatasetConfig(**{**base_dataset, "name": str(d)}))
        benchmark = BenchmarkConfig(
            mode=bench.get("mode", "single"),
            models=model_list,
            datasets=dataset_list,
            reproducibility=ReproducibilityConfig(**rep),
        )
        reports_raw = data.get("reports") or {}
        if isinstance(reports_raw, dict):
            vci = reports_raw.get("visual_case_ids")
            if vci is not None:
                vci = [str(x) for x in vci]
            reports = ReportsConfig(
                enabled=bool(reports_raw.get("enabled", False)),
                plugins=list(reports_raw.get("plugins") or []),
                save_comparison_meshes=bool(reports_raw.get("save_comparison_meshes", False)),
                comparison_mesh_subdir=str(
                    reports_raw.get("comparison_mesh_subdir") or "comparison_meshes"
                ),
                visual_case_ids=vci,
                visuals=list(reports_raw.get("visuals") or []),
            )
        else:
            reports = ReportsConfig()
        return cls(
            run=run,
            model=model,
            dataset=dataset,
            output=output,
            metrics=metrics,
            benchmark=benchmark,
            reports=reports,
        )

    @classmethod
    def _load_raw(cls, path: Path) -> dict[str, Any]:
        """Load a YAML or JSON file into a dict (empty dict if file is empty)."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                import json
                data = json.load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError(f"Config root must be a mapping, got {type(data).__name__}")
        return data

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load config from a single YAML or JSON file."""
        return cls.from_dict(cls._load_raw(Path(path)))

    @classmethod
    def load_merged(cls, *paths: str | Path) -> "Config":
        """Load multiple configs and deep-merge in order (later files override earlier)."""
        merged: dict[str, Any] = {}
        for p in paths:
            merged = deep_merge_dict(merged, cls._load_raw(Path(p)))
        return cls.from_dict(merged)

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply CLI-style overrides (e.g. run.device, model.checkpoint)."""
        for key, value in overrides.items():
            if "." not in key:
                continue
            parts = key.split(".")
            obj: Any = self
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                attr = parts[-1]
                current = getattr(obj, attr, None)
                if isinstance(current, bool) and isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                elif isinstance(current, int) and isinstance(value, str) and value.isdigit():
                    value = int(value)
                elif isinstance(current, float) and isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                setattr(obj, attr, value)


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
    *,
    base: str | Path | None = None,
) -> Config:
    """Load config from file and optionally apply overrides.

    If ``base`` is set, load ``base`` first then merge ``path`` on top (option A:
    shared inference defaults + benchmark overlay).
    """
    if base is not None:
        config = Config.load_merged(base, path)
    else:
        config = Config.load(path)
    if overrides:
        config.apply_overrides(overrides)
    return config
