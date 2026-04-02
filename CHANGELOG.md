<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2a0] - 2025-08-XX

### Added

- `physicsnemo.cfd.evaluation`: config-driven inference and benchmarking with
  metrics delegated to `physicsnemo.cfd.bench` (L2, area-weighted L2, forces,
  continuity/momentum residual L2 chain). Mesh bridge attaches GT and prediction
  fields using explicit VTK names from config (`output.mesh_field_names`,
  `output.ground_truth_mesh_field_names`, and volume equivalents).
- `physicsnemo.cfd.bench.metric_registry` for named metrics shared with evaluation.
- Example YAML under `workflows/evaluation_examples/`.
- CLI: `python -m physicsnemo.cfd.evaluation.benchmarks` (canonical evaluation entrypoint) and
  `python -m physicsnemo.cfd.evaluation.inference` (compatibility wrapper forwarding to benchmarks).

### Changed

- Benchmark metric registration for evaluation now uses bench-backed built-ins
  under `physicsnemo.cfd.evaluation.metrics.builtin` instead of duplicated
  NumPy-only helpers.
- Evaluation examples use **`evaluation_config.yaml`** and **`benchmarks.run`** as the primary
  workflow; `evaluation.inference` forwards to the benchmark engine; the benchmark runner can write
  **`inference_<model>_<case>.vtp|vtu`** when **`run.save_inference_mesh`** is true.
- Matrix benchmark settings live in the same **`evaluation_config.yaml`** (commented template); the
  separate **`benchmark_matrix.yaml`** overlay was removed from the examples folder.

### Deprecated

### Removed

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
