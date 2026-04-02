# Evaluation examples (PhysicsNeMo-CFD)

YAML examples and CLIs for **`physicsnemo.cfd.evaluation`**: single-case inference, benchmarking, and optional PNG reports. Install from this repo (`pip install -e .`) or `pip install nvidia-physicsnemo-cfd`, then run commands **from this directory** (or pass absolute paths to `--config`).

---

## Commands

**Inference** — one model, one case; writes `inference_<model>_<case>.vtp|vtu` under `run.output_dir`. With `reports.enabled: true` and `reports.visuals`, also writes a comparison mesh and PNGs (needs GT on the case; see [Reports and plots](#reports-and-plots)).

```bash
python -m physicsnemo.cfd.evaluation.inference.run --config inference_config.yaml
python -m physicsnemo.cfd.evaluation.inference.run --config inference_config.yaml --case-id run_1
# shorthand: python -m physicsnemo.cfd.evaluation.inference
```

**Benchmark** — metrics, JSON/CSV/HTML; **does not** run `reports.visuals` or save comparison VTK.

```bash
python -m physicsnemo.cfd.evaluation.benchmarks.run --base-config inference_config.yaml --config benchmark_matrix.yaml --case-id run_1
python -m physicsnemo.cfd.evaluation.benchmarks
```

**Merge order:** `--base-config` first, then `--config`, then optional CLI overrides (`run.device=cpu`, or `--run.device=cuda:1`). Inference supports `--base-config` the same way.

**Minimal surface-only example:** `surface_benchmark_minimal.yaml` (edit paths).

---

## Config files in this folder

| File | Use |
| ---- | --- |
| [`inference_config.yaml`](inference_config.yaml) | Shared defaults: `run`, `model`, `dataset`, `output`, `metrics`, optional `reports`, `benchmark.mode: single`. Use alone or as `--base-config`. |
| [`benchmark_matrix.yaml`](benchmark_matrix.yaml) | Overlay: `benchmark.mode: matrix`, `models[]`, `datasets[]` (Cartesian product; incompatible surface/volume pairs are skipped). |

---

## Config reference (merged YAML)

Keys are the same whether they come from one file or base + overlay.

| Section | Contents |
| ------- | -------- |
| **run** | `device`, `output_dir`, `seed`, `batch_size` |
| **model** | `name` (registered wrapper), `checkpoint`, `stats_path`, optional `inference_domain` (`surface` \| `volume`), `kwargs` |
| **dataset** | `name` (registered adapter), `root`, `split`, `case_ids` (`null` = all), `kwargs` (e.g. `align_ground_truth_to_model`, `inference_domain` for volume) |
| **output** | `mesh_field_names`, `ground_truth_mesh_field_names` (surface); `volume_mesh_field_names`, `ground_truth_volume_mesh_field_names` (volume) |
| **metrics** | List of metric names or `{ name: ..., ...kwargs }` — see [Metrics](#metrics) |
| **reports** | Optional PNG pipeline — see [Reports and plots](#reports-and-plots) |
| **benchmark** | `mode` (`single` \| `matrix`), `models` / `datasets`, `reproducibility` |

**Model notes:** GeoTransolver / Transolver on **volume** need `model.inference_domain: volume` and stats that match volume training (`global_stats.json` with `velocity`, `pressure_volume`, `turbulent_viscosity`, or `volume_fields_normalization.npz`). **Surface** GeoTransolver/Transolver needs surface stats (`pressure` + `shear_stress` in `global_stats.json` or `surface_fields_normalization.npz`). DoMINO uses `domino_config` for volume.

**Custom code:** [Custom models](#custom-models-adding-a-new-wrapper) · [Custom datasets](#custom-datasets-adding-a-new-adapter) · [Baseline stubs](#baseline-model-stubs-surface_baseline-volume_baseline)

---

## Metrics

Registered names (or dicts with `name` + kwargs). Example list from `inference_config.yaml`:

```yaml
metrics:
  - l2_pressure
  - l2_shear_stress
  - l2_pressure_area_weighted   # alias: area_wt_l2_pressure
  - l2_pressure_volume          # needs volume case + output maps
  - l2_velocity
  - l2_turbulent_viscosity
  - drag_error
  - lift_error
  - continuity_residual_l2      # volume: needs velocity on volume mesh
  - momentum_residual_l2
```

| Name | Meaning |
| ---- | ------- |
| `l2_pressure`, `l2_shear_stress` | Surface L2 |
| `l2_pressure_area_weighted` | Area-weighted L2 pressure (`area_wt_l2_pressure`) |
| `l2_pressure_volume` | Volume pressure (`pressure_volume`) |
| `l2_velocity`, `l2_turbulent_viscosity` | Volume / surface depending on case |
| `drag_error`, `lift_error` | Coefficient errors |
| `continuity_residual_l2`, `momentum_residual_l2` | Volume residuals |

---

## Reports and plots

### What runs where

| CLI | Metrics | `reports.visuals` (PNGs) | Comparison VTK |
| --- | ------- | -------------------------- | ---------------- |
| **Inference** | — | Yes, if `reports.enabled` and `visuals` non-empty | Saved if `save_comparison_meshes` **or** visuals enabled |
| **Benchmark** | Yes | No | Not written by engine |

Built-in **visual** names: `field_comparison_surface` (surface GT vs pred), `plot_fields_volume` (volume scalars). Register more: `physicsnemo.cfd.evaluation.reports.register_visual`.

**Outputs:** PNGs under `{run.output_dir}/visuals/`; manifest `report_plugins_manifest.json`; comparison mesh `{run.output_dir}/{comparison_mesh_subdir}/{case_id}_comparison.vtp|vtu`.

**Inference writes two meshes:** (1) `inference_<model>_<case>.vtp|vtu` — **predictions only**; (2) `comparison_meshes/<case>_comparison.*` — **GT + pred** (same as `build_comparison_mesh` for metrics/plots).

### `reports` YAML shape

```yaml
reports:
  enabled: true
  save_comparison_meshes: false
  comparison_mesh_subdir: comparison_meshes
  visuals:
    - field_comparison_surface
    - name: field_comparison_surface
      case_ids: ["run_1"]       # must match the case you run (--case-id)
      canonical_keys: [pressure, shear_stress]
      view: xy
```

| Key | Role |
| --- | ---- |
| `enabled` | If true, run registered visuals (inference calls this automatically). |
| `save_comparison_meshes` | Save comparison VTK even with no `visuals` (inference only). |
| `comparison_mesh_subdir` | Subfolder for `*_comparison.vtp|vtu`. |
| `visuals` | Ordered list: string names or `{ name: ..., ...kwargs }`. |
| `plugins` | Legacy manifest hooks (`ReportsConfig`). |

### Field selection (built-ins)

- **`field_comparison_surface`** — Uses `canonical_keys` (default `pressure`) and `output.ground_truth_mesh_field_names` / `mesh_field_names`.
- **`plot_fields_volume`** — `fields` = VTK array names, or default all `output.volume_mesh_field_names` values.

### Line plots

There is **no** built-in **`reports.visuals`** entry for **`plot_line`** (needs polyline extracts + `field_true` / `field_pred`). Options: **register** a custom visual that calls `physicsnemo.cfd.bench.visualization.utils.plot_line`, or use [`workflows/bench_example`](../bench_example/README.md) scripts.

### Other plot paths

- **Bench example:** run inference to get VTK, then `generate_surface_benchmarks.py` / `generate_volume_benchmarks.py` with a matching field map.
- **Programmatic:** `bench.visualization.utils` or `run_optional_report_plugins(config, results, output_dir)` with `per_case[].comparison_mesh_path` if you build results yourself.

### Troubleshooting

- **`case_ids`** in a visual must include the case you run — use `--case-id run_1` if the visual lists `case_ids: ["run_1"]`, or omit `case_ids`.
- Use an **editable install** or **`PYTHONPATH`** to this repo so inference includes the visualization step (not an old wheel).
- Check **`report_plugins_manifest.json`** → `visual_errors` (headless PyVista often needs **xvfb**).

---

## Baseline model stubs (`surface_baseline`, `volume_baseline`)

Optional **no-checkpoint** wrappers for smoke tests, layout checks, and **matrix skip** behavior. Not real physics: zeros / placeholders with correct canonical keys. Set `model.name` explicitly; use `checkpoint: ""`, `stats_path: ""`.

```yaml
model:
  name: "surface_baseline"   # or volume_baseline
  checkpoint: ""
  stats_path: ""
dataset:
  name: drivaerml
  root: /path/to/data
  # kwargs: { inference_domain: volume }   # for volume_baseline
```

---

## Custom models (adding a new wrapper)

1. Subclass **`CFDModel`** (`physicsnemo.cfd.evaluation.inference.model_registry`) under `physicsnemo/cfd/evaluation/inference/wrappers/`.
2. Implement `INFERENCE_DOMAIN`, `OUTPUT_LOCATION`, `load`, `prepare_inputs`, `predict`, `decode_outputs` (canonical keys: surface `pressure`, `shear_stress`; volume `pressure_volume`, `turbulent_viscosity`, `velocity`).
3. **`register_model("my_model", MyWrapper)`** in `wrappers/__init__.py`.

Reference: `wrappers/fignet/wrapper.py`, `geotransolver/wrapper.py`, `surface_baseline.py`, …

---

## Custom datasets (adding a new adapter)

1. Subclass **`DatasetAdapter`** under `physicsnemo/cfd/evaluation/datasets/adapters/`.
2. Implement `list_cases`, `load_case` → **`CanonicalCase`** (`mesh_path`, `mesh_type`, `ground_truth`, `inference_domain`), optional **`inference_domain_from_kwargs`** for matrix skips.
3. **`register_adapter("my_dataset", MyAdapter)`** in `adapters/__init__.py`.

Reference: `adapters/drivaerml.py`, `ahmed.py`. Helpers: `datasets/vtk_ground_truth.py`, `gt_alignment.py`.

---

## Canonical types (`physicsnemo.cfd.evaluation.datasets.schema`)

- **`CanonicalCase`** — `case_id`, `mesh_path`, `mesh_type`, `ground_truth`, `metadata`, `inference_domain`.
- **Predictions** (model output) — surface: `pressure`, `shear_stress`; volume: `pressure_volume`, `turbulent_viscosity`, `velocity`.

---

## Package layout (repo root)

| Path | Role |
| ---- | ---- |
| `physicsnemo/cfd/evaluation/config.py` | Config, `load_config`, `Config.load_merged` |
| `physicsnemo/cfd/evaluation/inference/` | Model registry, `wrappers/` |
| `physicsnemo/cfd/evaluation/datasets/` | Adapters, `CanonicalCase`, GT helpers |
| `physicsnemo/cfd/evaluation/metrics/` | Registry, `mesh_bridge.py`, bench-backed metrics |
| `physicsnemo/cfd/evaluation/reports/` | `register_visual`, built-in plots |
| `physicsnemo/cfd/evaluation/benchmarks/` | Engine, JSON/CSV/HTML reports |
| `physicsnemo/cfd/bench/` | Shared algorithms (L2, forces, …) |

---

