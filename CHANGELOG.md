<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3a0] - 2026-XX-YY

### Added

- **Agent skills:** `skills/` now ships three agent skills for the
  benchmarking workflow — **`physicsnemo-cfd-create-model-wrapper`**,
  **`physicsnemo-cfd-create-dataset-adapter`**, and
  **`physicsnemo-cfd-create-custom-metric`** — each with a `SKILL.md` and
  `evals/` cases to guide adding a new `CFDModel`, `DatasetAdapter`, or
  metric.

### Changed

### Deprecated

### Removed

### Fixed

### Security

### Dependencies

## [0.0.2] - 2026-06-09

### Added

- **`physicsnemo.cfd.evaluation.assets`:** first-party **`Package`** helper for
  **`hf://`**, **`s3://`**, and local directories (optional extra
  **`evaluation-hf`**: `huggingface_hub`, `fsspec`); **`resolve_model_assets`**
  in the benchmark engine; **`ModelConfig.package`**, **`checkpoint_relpath`**,
  **`stats_relpath`**; **`register_default_asset`** / **`AssetSpec`** for
  default Hub layouts; metrics-cache fingerprint **`asset_identity`** for
  stable keys when using remote packages; baseline wrappers set
  **`REQUIRES_REMOTE_ASSETS = False`**. **`ngc://`** is reserved (not
  implemented).
- **`output.surface_interpolate_point_to_cell_for_metrics`:** optional kNN-IDW
  promotion of surface point fields to cell centers before metrics
  (``interpolate_point_data_to_cell_centers`` in
  ``postprocessing_tools/interpolation``), so point-native models (e.g. XmGN,
  FiGNet) can report cell-based L2, drag, and lift. Tunables:
  ``output.surface_metrics_idw_k``, ``output.surface_metrics_idw_device``.
- **`run.fail_on_all_skipped` / `run.fail_on_any_metric_nan`:** optional
  benchmark exit policy; raises `BenchmarkPolicyError` (Hydra `main.py` and
  `python -m physicsnemo.cfd.evaluation.benchmarks.run` exit with code 1).
- **`physicsnemo.cfd.evaluation.benchmarks.hydra_utils`:** shared Hydra →
  `Config` dict conversion for tests and `workflows/benchmarking/main.py`.
- **CI:** `.github/workflows/ci-tests.yml` runs `pytest` on `test/ci_tests/`
  (including `test_benchmark_workflow.py`, which composes Hydra configs under
  `workflows/benchmarking/conf/`).
- **Multi-GPU benchmarks:** `run_benchmark` / Hydra `main.py` can run under
  `torchrun` with PhysicsNeMo `DistributedManager` — per-rank case sharding,
  gather/merge on rank 0, and optional `run.distributed` (default true) to
  disable sharding for debugging.
- **Tests:** `test/ci_tests/test_distributed_utils.py` covers merge/shard
  helpers and `_case_ids_for_run`.
- **Tests:** `test/ci_tests/test_benchmark_workflow.py` composes workflow YAML
  and loads `Config`.
- **Metrics cache:** optional `run.metrics_cache` (`enabled`, `path`) stores
  per-case scalar metrics on disk so repeat benchmark runs can skip VTK I/O and
  inference for unchanged configs; visualization is unchanged.
- `physicsnemo.cfd.evaluation`: config-driven inference and benchmarking with
  metrics delegated to `physicsnemo.cfd.postprocessing_tools` (L2,
  area-weighted L2, forces, continuity/momentum residual L2 chain). Mesh
  bridge attaches GT and prediction fields using explicit VTK names from
  config (`output.mesh_field_names`, `output.ground_truth_mesh_field_names`,
  and volume equivalents).
- `physicsnemo.cfd.postprocessing_tools.metric_registry` for named metrics
  shared with evaluation.
- Example Hydra configs under `workflows/benchmarking/conf/`, including matrix
  templates `config_matrix_surface_custom.yaml` /
  `config_matrix_surface_hf.yaml` and volume equivalents (multiple models ×
  dataset blocks; **`_custom`** = explicit paths, **`_hf`** = Hugging Face
  **`builtin_packages`**).
- Library CLIs remain for scripting:
  `python -m physicsnemo.cfd.evaluation.benchmarks` /
  `python -m physicsnemo.cfd.evaluation.inference` (flat YAML path, no
  `${...}` interpolation).

### Changed

- **STL resolution in run directories:** ``_find_stl_in_dir``
  (``evaluation.models.common_wrapper_utils.vtk_datapipe_io``, shared by
  Transolver / GeoTransolver / DoMINO-style VTK helpers) tries
  ``drivaer_{run_idx}_single_solid.stl`` **before** ``drivaer_{run_idx}.stl``.
  If a directory contains **both** (e.g. mixed surface/volume exports),
  geometry now follows that order; legacy DoMINO-only helpers used the
  reverse. Remove or rename one file if you need the other mesh.
- **Inference dtype / autocast:** FiGNet, XmGN, and DoMINO support model
  kwarg ``cuda_bf16_autocast`` (default ``true``; set ``false`` for full
  fp32). Transolver and GeoTransolver use the same kwarg with default
  ``false`` (fp32 forwards); set ``true`` for CUDA bf16 autocast to match the
  other wrappers' default inference style.
- **Package layout:** model wrappers, ``model_registry``, and
  ``common_wrapper_utils`` moved to **`physicsnemo.cfd.evaluation.models`**.
  **`physicsnemo.cfd.evaluation.inference`** now holds the flat-YAML CLI
  (`run.py`, `__main__.py`) and **`progress`** (`log_inference`). The
  **`inference`** package still re-exports **`CFDModel`**,
  **`get_model_wrapper`**, etc., for backward compatibility; new code should
  import from **`evaluation.models`**.
- **Built-in benchmark packages:**
  **`physicsnemo.cfd.evaluation.assets.builtin_packages`** registers default
  **`hf://`** roots and relative checkpoint/stats paths for
  **geotransolver**, **transolver**, **xmgn**, and **fignet** (per-model
  roots such as **`GEOTRANSOLVER_PACKAGE_ROOT`** in **`builtin_packages`**).
  Omit **`checkpoint`** / **`stats_path`** to use those Hub layouts; when
  both are set, they override the built-in package. **domino** also resolves
  **`domino_config`** from the same package via
  :class:`~physicsnemo.cfd.evaluation.assets.registry.AssetSpec`
  **`extra_resolve_relpaths`**. Edit each model’s root when repos differ.
- **Surface comparison mesh:** ``build_comparison_mesh`` respects
  ``CanonicalCase.mesh_type`` — point GT/pred (e.g. ``xmgn`` with
  ``align_ground_truth_to_model``) no longer forces
  ``point_data_to_cell_data``, fixing length mismatches vs cell counts.
- **Benchmarking workflow layout:** legacy **`workflows/bench_example`** moved
  to **`workflows/deprecated/bench_example`** (superseded by
  **`workflows/benchmarking/`**).
- **`benchmark.reproducibility.log_env`:** default is now **`false`** (was
  **`true`**) to avoid writing full `os.environ` to `env.json` unless
  explicitly enabled in YAML.
- **Breaking:** `physicsnemo.cfd.bench` was renamed to
  **`physicsnemo.cfd.postprocessing_tools`** (same submodules: `metrics`,
  `visualization`, `geometry`, `interpolation`, `metric_registry`).
- **`reports.visual_case_ids`**: optional list limiting which cases get an
  in-memory comparison mesh for PNG visuals (default: all cases). Same list
  is the default **`case_ids`** for visuals that omit it; per-visual
  **`case_ids`** still overrides.
- DoMINO NIM helper `call_domino_nim` and `launch_local_domino_nim.sh` live
  under **`physicsnemo.cfd.evaluation.nims`** (removed top-level
  **`physicsnemo.cfd.inference`** package).
- **GeoTransolver** checkpoint load uses ``trusted_torch_load_context``
  (trusted checkpoints).
- Matrix benchmark domain-mismatch skip logs via ``log_dataset`` instead of
  ``print``.
- Benchmark metric registration for evaluation now uses
  postprocessing_tools-backed built-ins under
  `physicsnemo.cfd.evaluation.metrics.builtin` instead of duplicated
  NumPy-only helpers.
- Evaluation **workflow docs** center on **`workflows/benchmarking/main.py`**
  (Hydra) and **`conf/config_matrix_surface_*.yaml`** /
  **`config_matrix_volume_*.yaml`**;
  **`python -m physicsnemo.cfd.evaluation.inference`** still forwards to the
  benchmark engine; **`inference_<model>_<case>.vtp|vtu`** when
  **`run.save_inference_mesh`** is true.
- Matrix benchmark settings are edited in the **`conf/*.yaml`** files; the
  old **`benchmark_matrix.yaml`** overlay was removed earlier from the
  examples folder.

### Removed

- **`workflows/benchmarking/evaluation_config.yaml`** — the workflow is driven
  by Hydra **`conf/config_matrix_*.yaml`** and **`main.py`** (no separate
  single-model **`config_surface`** / **`config_volume`** in `conf/`).

### Fixed

- **Surface comparison mesh:** ``build_comparison_mesh`` infers point vs cell
  attachment from GT length when it matches exactly one of ``mesh.n_points``
  or ``mesh.n_cells``, so datasets that register ``mesh_type: cell`` but
  supply point-sized fields (e.g. some DrivAerML paths) no longer fail with
  ``expected N cell values, got ...`` mismatches.
- Fixed a bug in GPU kernel for gradient computation.

### Dependencies

- Improved the dependency handling.
- Added `httpx` dependency.
- Bumped `warp-lang` to `warp-lang>=1.13.0` (in the `gpu` extra) to
  track upstream's removal of the `wp.context` namespace. PhysicsNeMo's
  matching `wp.Device` migration (used by SDF / kNN-IDW interpolation) is
  picked up via the temporarily git-pinned `nvidia-physicsnemo` revision.

## [0.0.1] - 2025-06-15

### Added

- Initial working release of benchmarking tools, hybrid initialization example,
  and inference tools for the DoMINO NVIDIA Inference Microservice (NIM).
