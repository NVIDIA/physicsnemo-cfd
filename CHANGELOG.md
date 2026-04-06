<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2a0] - 2025-08-XX

### Added

- **Hydra workflow** for evaluation examples: `workflows/evaluation_examples/main.py` with
  `conf/config_surface.yaml` (default) and `conf/config_volume.yaml` (`hydra-core` dependency), consistent
  with other workflows (see **Removed** for dropped flat example YAML).
- **Multi-GPU benchmarks:** `run_benchmark` / Hydra `main.py` can run under `torchrun` with PhysicsNeMo `DistributedManager` — per-rank case sharding, gather/merge on rank 0, and optional `run.distributed` (default true) to disable sharding for debugging.
- **Tests:** `test/ci_tests/test_distributed_utils.py` covers merge/shard helpers and `_case_ids_for_run`.
- **Metrics cache:** optional `run.metrics_cache` (`enabled`, `path`) stores per-case scalar metrics on disk
  so repeat benchmark runs can skip VTK I/O and inference for unchanged configs; visualization is unchanged.
- `physicsnemo.cfd.evaluation`: config-driven inference and benchmarking with
  metrics delegated to `physicsnemo.cfd.postprocessing_tools` (L2, area-weighted L2, forces,
  continuity/momentum residual L2 chain). Mesh bridge attaches GT and prediction
  fields using explicit VTK names from config (`output.mesh_field_names`,
  `output.ground_truth_mesh_field_names`, and volume equivalents).
- `physicsnemo.cfd.postprocessing_tools.metric_registry` for named metrics shared with evaluation.
- Example Hydra configs under `workflows/evaluation_examples/conf/`.
- Library CLIs remain for scripting: `python -m physicsnemo.cfd.evaluation.benchmarks` /
  `python -m physicsnemo.cfd.evaluation.inference` (flat YAML path, no `${...}` interpolation).

### Changed

- **Breaking:** `physicsnemo.cfd.bench` was renamed to **`physicsnemo.cfd.postprocessing_tools`**
  (same submodules: `metrics`, `visualization`, `geometry`, `interpolation`, `metric_registry`).
- **`reports.visual_case_ids`**: optional list limiting which cases get an in-memory comparison mesh for
  PNG visuals (default: all cases). Same list is the default **`case_ids`** for visuals that omit it;
  per-visual **`case_ids`** still overrides.
- DoMINO NIM helper `call_domino_nim` and `launch_local_domino_nim.sh` live under
  **`physicsnemo.cfd.evaluation.nims`** (removed top-level **`physicsnemo.cfd.inference`** package).
- **GeoTransolver** checkpoint load uses ``trusted_torch_load_context`` (trusted checkpoints).
- Matrix benchmark domain-mismatch skip logs via ``log_dataset`` instead of ``print``.
- Benchmark metric registration for evaluation now uses postprocessing_tools-backed built-ins
  under `physicsnemo.cfd.evaluation.metrics.builtin` instead of duplicated
  NumPy-only helpers.
- Evaluation **workflow docs** center on **`workflows/evaluation_examples/main.py`** (Hydra) and
  **`conf/config_surface.yaml`** / **`conf/config_volume.yaml`**; `evaluation.inference` still forwards
  to the benchmark engine; **`inference_<model>_<case>.vtp|vtu`** when **`run.save_inference_mesh`** is true.
- Matrix benchmark settings are edited in the **`conf/*.yaml`** files; the old **`benchmark_matrix.yaml`**
  overlay was removed earlier from the examples folder.

### Deprecated

### Removed

- **`workflows/evaluation_examples/evaluation_config.yaml`** — evaluation examples are driven by Hydra **`conf/config_surface.yaml`** / **`config_volume.yaml`** and **`main.py`**.

### Fixed

- Fixed a bug in GPU kernel for gradient computation.

### Security

### Dependencies

- Improved the dependency handling.
- Added `httpx` dependency.

## [0.0.1] - 2025-06-15

### Added

- Initial working release of benchmarking tools, hybrid initialization example,
  and inference tools for the DoMINO NVIDIA Inference Microservice (NIM).
