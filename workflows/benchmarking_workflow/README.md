# Model evaluation and benchmarking

This is an opinionated workflow to help you get started with evaluating and benchmarking pretrained models.  It stitches together the model inference, calcuating built-in metrics and generating artifacts for analysis and visualization. We use a configuration based approach to allow you to extend this workflow to add your datasets, models, custom metrics etc.

# Running OOB  
We use configuration files to setup the model evaluation pipeline. These configs are under **`conf/`** for surface, volume evaluations for a model and a matrix run for comparing multiple models: **[`config_surface.yaml`](conf/config_surface.yaml)** (default), **[`config_volume.yaml`](conf/config_volume.yaml)**, and matrix examples **[`config_matrix_surface.yaml`](conf/config_matrix_surface.yaml)** / **[`config_matrix_volume.yaml`](conf/config_matrix_volume.yaml)** (several models × several dataset entries). 

As a first step, you can this benchmarking example (link) to do an out of the box run.

```bash
# Surface benchmark (VTP / wall metrics, default config)
python main.py

# Volume benchmark (VTU / volume metrics)
python main.py --config-name=config_volume

# Matrix: multiple models × multiple dataset blocks (see conf/config_matrix_*.yaml)
python main.py --config-name=config_matrix_surface
python main.py --config-name=config_matrix_volume

# Hydra overrides (examples)
python main.py case_id=run_1 run.device=cuda:0
python main.py --config-name=config_volume run.output_dir=my_volume_run
```

**Multi-GPU (optional):** 

```bash
torchrun --standalone --nproc_per_node=4 main.py
# or: torchrun --standalone --nproc_per_node=4 main.py --config-name=config_volume
```

Cases are split across GPUs (`cases[rank::world_size]`). Rank 0 writes `benchmark_results.*` and optional artifacts; all ranks return the merged result list. Set `run.distributed: false` only if debugging (each rank would run the full case list).

**Matrix mode:** use **[`conf/config_matrix_surface.yaml`](conf/config_matrix_surface.yaml)** or **[`conf/config_matrix_volume.yaml`](conf/config_matrix_volume.yaml)** as templates, or set `benchmark.mode: matrix` in any config and fill **`benchmark.models`** × **`benchmark.datasets`**. The product runs every model on every dataset entry; incompatible surface/volume pairs are skipped automatically. Edit checkpoint paths for your layout; comment out any **`- name:`** model block you do not want to run.

# How to Extend (Add Your Own Data/Models)
To extend this workflow for your own custom evaluation or benchmarking, you will likely want to customize three things:

## Custom models (adding a new wrapper)

1. Subclass **`CFDModel`** (`physicsnemo.cfd.evaluation.inference.model_registry`) under `physicsnemo/cfd/evaluation/inference/wrappers/`.
2. Implement `INFERENCE_DOMAIN`, `OUTPUT_LOCATION`, `load`, `prepare_inputs`, `predict`, `decode_outputs`.
3. **`register_model("my_model", MyWrapper)`** in `wrappers/__init__.py`.

## Custom datasets (adding a new adapter)

1. Subclass **`DatasetAdapter`** under `physicsnemo/cfd/evaluation/datasets/adapters/`.
2. Implement `list_cases`, `load_case` → **`CanonicalCase`**.
3. **`register_adapter("my_dataset", MyAdapter)`** in `adapters/__init__.py`.

## Canonical types (`physicsnemo.cfd.evaluation.datasets.schema`)

- **`CanonicalCase`** — `case_id`, `mesh_path`, `mesh_type`, `ground_truth`, `metadata`, `inference_domain`.
- **Predictions** — surface: `pressure`, `shear_stress`; volume: `pressure`, `turbulent_viscosity`, `velocity`.
---

## Config files

| File | Role |
| ---- | ----- |
| [`conf/config_surface.yaml`](conf/config_surface.yaml) | **Surface**, `benchmark.mode: matrix`, **GeoTransolver** × **drivaerml**. Top-level **`case_id: [run_1, run_11]`** (no per-dataset `case_ids`). **`run.device`**: `cuda:1`. **`run.output_dir`**: **`benchmark_results_surface`**. `reports.enabled: false` (visuals listed for copy/paste when enabling reports). |
| [`conf/config_volume.yaml`](conf/config_volume.yaml) | **Volume**, `benchmark.mode: matrix`, **GeoTransolver** × **drivaerml**. **`case_id: [run_1, run_11]`**. **`run.device`**: `cuda:0`. **`run.output_dir`**: **`benchmark_results_volume`**. |
| [`conf/config_matrix_surface.yaml`](conf/config_matrix_surface.yaml) | **Matrix surface:** **GeoTransolver, Transolver, XMGN, FigNet, DoMINO** × **one** drivaerml row; **`case_id: null`** (all cases unless overridden). Checkpoints under `.../benchmark_models/<model>_drivaerml_surface_checkpoint/`. DoMINO **`kwargs.domino_config`** points at a YAML next to the checkpoint. **`run.output_dir`**: **`benchmark_results_matrix_surface`**. Optional commented baseline stub at bottom of **`benchmark.models`**. |
| [`conf/config_matrix_volume.yaml`](conf/config_matrix_volume.yaml) | **Matrix volume:** same five model names in **volume** mode × **one** drivaerml row; **`case_id: null`**. Paths use `.../_drivaerml_volume_checkpoint/`. **`run.output_dir`**: **`benchmark_results_matrix_volume`**. |

---

## Config reference (YAML keys)

| Section | Contents |
| ------- | -------- |
| **run** | `device`, `output_dir`, `seed`, `batch_size`, **`save_inference_mesh`**, **`distributed`** (default true: shard cases under `torchrun`), optional **`metrics_cache`** (see below) |
| **model** / **benchmark.models** | `name`, `checkpoint`, `stats_path`, `inference_domain` (`surface` \| `volume`), `kwargs` |
| **dataset** / **benchmark.datasets** | `name`, `root`, `split`, optional per-dataset `case_ids` (`null` = all), `kwargs` |
| **output** | `mesh_field_names`, `ground_truth_mesh_field_names`; `volume_mesh_field_names`, `ground_truth_volume_mesh_field_names`; **`streamlines_vector_canonical`** (default `velocity`) |
| **metrics** | Metric names or `{ name: ..., ...kwargs }` — see [Metrics](#metrics) |
| **reports** | Optional PNG pipeline — see [Reports and plots](#reports-and-plots) |
| **benchmark** | `mode` (`single` \| `matrix`), `models` / `datasets`, `reproducibility` |

**`case_id`:** optional top-level Hydra key: `null` uses each dataset's `case_ids` (or all adapter cases); a **string** runs one case on every dataset; a **list** runs those cases on every dataset in matrix mode (preferred over repeating `case_ids` on each dataset). CLI examples: `case_id=run_1`, `'case_id=[run_1,run_11]'`.

**Metrics cache (optional):** Under `run.metrics_cache`, `enabled: true` stores per-case scalars under `path` (resolved; default pattern `${run.output_dir}/metrics_cache`). Delete that directory for a full recompute. **Plots and meshes are not cached.**

**Model notes:** GeoTransolver / Transolver **volume** needs matching stats (`global_stats.json` with `velocity`, `pressure`, `turbulent_viscosity`, or `volume_fields_normalization.npz`). **Surface** needs surface stats. DoMINO uses `domino_config` for volume.

**Custom code:** [Custom models](#custom-models-adding-a-new-wrapper) · [Custom datasets](#custom-datasets-adding-a-new-adapter) · [Baseline stubs](#baseline-model-stubs-surface_baseline-volume_baseline)

---

## Advanced

The Python API **`physicsnemo.cfd.evaluation.benchmarks.engine.run_benchmark`** and **`Config.from_dict`** remain for scripts and tests. The modules **`python -m physicsnemo.cfd.evaluation.benchmarks.run`** and **`evaluation.inference`** accept a YAML/JSON path **without** Hydra interpolation — use only if you supply a **flat** file (no `${...}`) or materialized values.

---

## Metrics

Registered names (or dicts with `name` + kwargs). Examples match **`conf/config_surface.yaml`** / **`config_volume.yaml`**:

```yaml
# Surface-oriented
metrics:
  - l2_pressure
  - l2_shear_stress
  - l2_pressure_area_weighted
  - drag
  - lift

# Volume-oriented (see config_volume.yaml)
metrics:
  - l2_pressure
  - l2_turbulent_viscosity
  - l2_velocity
  - continuity_residual_l2
  - momentum_residual_l2
```

| Name | Meaning |
| ---- | ------- |
| `l2_pressure`, `l2_shear_stress` | Surface L2 |
| `l2_pressure_area_weighted` | Area-weighted L2 pressure (`area_wt_l2_pressure`) |
| `l2_pressure` (volume) | Volume pressure (domain-scoped, same name as surface) |
| `l2_velocity`, `l2_turbulent_viscosity` | Volume / surface depending on case |
| `drag`, `lift` | Drag / lift coefficient errors (expands to `drag_error`, `drag_true`, `drag_pred`, etc.) |
| `continuity_residual_l2`, `momentum_residual_l2` | Volume residuals |

---

## Reports and plots

### What runs where

The benchmark engine writes **`benchmark_results.json` / `.csv` / `.html`**, then optional **`reports.visuals`** PNGs (when `reports.enabled` and `visuals` is non-empty). Comparison VTK on disk exists only if **`reports.save_comparison_meshes: true`**. **`run.save_inference_mesh: true`** adds prediction-only VTK per case.

| Name | Role |
| ---- | ---- |
| `field_comparison_surface` | Surface GT vs pred (`plot_field_comparisons`) |
| `plot_fields_volume` | Volume scalars (`plot_fields`) |
| `line_plot` | GT vs pred along `plot_coord` |
| `design_scatter` | Pred vs true scatter + R²; needs **`pairs`** matching `per_case[].metrics` |
| `design_trend` | Trend vs index; same **`pairs`**, optional **`idx_key`** |
| `projections_hexbin` | Hexbin; needs **`mesh_paths`**, **`field`**, **`direction`** |
| `streamlines_comparison` | Uses **`output.streamlines_vector_canonical`** or **`canonical_key`** |

Register more: `physicsnemo.cfd.evaluation.reports.register_visual`.

**Outputs:** PNGs under `{run.output_dir}/visuals/`; manifest `report_plugins_manifest.json`; optional comparison mesh `{run.output_dir}/{comparison_mesh_subdir}/`.

### `reports` YAML shape

```yaml
reports:
  enabled: true
  save_comparison_meshes: false
  comparison_mesh_subdir: comparison_meshes
  visuals:
    - field_comparison_surface
    - name: field_comparison_surface
      case_ids: ["run_1"]
      canonical_keys: [pressure, shear_stress]
      view: xy
```

| Key | Role |
| --- | --- |
| `enabled` | If true, run registered visuals after metrics/tables. |
| `save_comparison_meshes` | Save comparison VTK to disk. |
| `visual_case_ids` | Optional; limits in-memory meshes for PNGs; default `case_ids` for visuals that omit it. |
| `visuals` | Ordered list: string names or `{ name: ..., ...kwargs }`. |

### Bench example ↔ evaluation mapping

| `workflows/bench_example` / `postprocessing_tools.visualization.utils` | Evaluation visual name |
| --------------------------------------------------------- | ------------------------ |
| `plot_field_comparisons` | `field_comparison_surface` |
| `plot_fields` | `plot_fields_volume` |
| `plot_line` | `line_plot` |
| `plot_design_scatter` | `design_scatter` |
| `plot_design_trend` | `design_trend` |
| `plot_projections_hexbin` | `projections_hexbin` |
| `plot_streamlines` | `streamlines_comparison` |

### Line plots

**`line_plot`** uses **`canonical_key`** (surface: `pressure`, `shear_stress`; volume: `pressure`, `velocity`, …) and **`plot_coord`**. Centerline-style strip plots: see [`workflows/bench_example`](../bench_example/README.md) or a custom `register_visual`.

### Troubleshooting

- **`case_ids` in a visual** must cover the cases you evaluate; set top-level **`case_id`** (string or list) or per-visual **`case_ids`** consistently.
- Headless PNG: **`report_plugins_manifest.json`** → `visual_errors`; often need **xvfb** (see `setup.sh`).
- Use an **editable install** so the installed package includes your local `evaluation` / `postprocessing_tools` changes.

---

## Baseline model stubs (`surface_baseline`, `volume_baseline`)

Smoke-test wrappers: **`checkpoint: ""`**, **`stats_path: ""`**. Set **`model.name`** in **`benchmark.models`** (matrix) or top-level **`model`** (single).

```yaml
model:
  name: "surface_baseline"
  checkpoint: ""
  stats_path: ""
dataset:
  name: drivaerml
  root: /path/to/data
```

---

## Package layout (repo root)

| Path | Role |
| ---- | ---- |
| `physicsnemo/cfd/evaluation/config.py` | `Config`, `load_config` |
| `physicsnemo/cfd/evaluation/benchmarks/` | Engine, JSON/CSV/HTML reports |
| `physicsnemo/cfd/evaluation/reports/` | `register_visual`, built-in plots |
| `physicsnemo/cfd/postprocessing_tools/` | Shared metrics / visualization algorithms |
