# Evaluation examples (PhysicsNeMo-CFD)

YAML and the **benchmark** CLI drive all evaluation: **metrics**, tabular outputs (JSON/CSV/HTML), optional **PNG visuals**, and optional VTK (`run.save_inference_mesh`, `reports.save_comparison_meshes`). Everything is configured in **[`evaluation_config.yaml`](evaluation_config.yaml)** — `benchmark.mode: single` uses the top-level `model` and `dataset`; **`benchmark.mode: matrix`** uses `benchmark.models` × `benchmark.datasets` (see the commented template at the bottom of that file). Install from this repo (`pip install -e .`) or `pip install nvidia-physicsnemo-cfd`, then run commands **from this directory** (or pass absolute paths to `--config`).

---

## Commands (canonical)

```bash
# One config: metrics + tables + optional visuals (see evaluation_config.yaml)
python -m physicsnemo.cfd.evaluation.benchmarks.run --config evaluation_config.yaml
python -m physicsnemo.cfd.evaluation.benchmarks.run --config evaluation_config.yaml --case-id run_1
# shorthand:
python -m physicsnemo.cfd.evaluation.benchmarks
```

**Matrix:** edit [`evaluation_config.yaml`](evaluation_config.yaml) — set `benchmark.mode: matrix` and fill `benchmark.models` / `benchmark.datasets` (template is commented at the bottom of the file). Then run the same command as above. Incompatible surface/volume pairs are skipped.

**Optional second config:** `--base-config` then `--config` still works if you prefer splitting shared defaults from an overlay; one file is enough for most workflows.

**Merge order** when using two files: `--base-config` first, then `--config`, then optional CLI overrides (`run.device=cpu`, or `--run.device=cuda:1`).

**Compatibility:** `python -m physicsnemo.cfd.evaluation.inference` forwards to the same benchmark engine (defaults to **one case** = first listed case when `--case-id` is omitted). Prefer the `benchmarks.run` command above.

---

## Config file

| File | Use |
| ---- | --- |
| [`evaluation_config.yaml`](evaluation_config.yaml) | **Full evaluation config:** `run`, `model`, `dataset`, `output`, `metrics`, optional `reports`, and `benchmark` (`mode: single` or `mode: matrix` with `models` / `datasets`). Matrix example is in comments at the bottom of the file. |

---

## Config reference (merged YAML)

Keys are the same whether they come from one file or base + overlay.

| Section | Contents |
| ------- | -------- |
| **run** | `device`, `output_dir`, `seed`, `batch_size`, **`save_inference_mesh`** (if `true`, benchmark writes `inference_<model>_<case>.vtp\|vtu` per case) |
| **model** | `name` (registered wrapper), `checkpoint`, `stats_path`, optional `inference_domain` (`surface` \| `volume`), `kwargs` |
| **dataset** | `name` (registered adapter), `root`, `split`, `case_ids` (`null` = all), `kwargs` (e.g. `align_ground_truth_to_model`, `inference_domain` for volume) |
| **output** | `mesh_field_names`, `ground_truth_mesh_field_names` (surface); `volume_mesh_field_names`, `ground_truth_volume_mesh_field_names` (volume); optional **`streamlines_vector_canonical`** (default `velocity`) for `streamlines_comparison` |
| **metrics** | List of metric names or `{ name: ..., ...kwargs }` — see [Metrics](#metrics) |
| **reports** | Optional PNG pipeline — see [Reports and plots](#reports-and-plots) |
| **benchmark** | `mode` (`single` \| `matrix`), `models` / `datasets`, `reproducibility` |

**Model notes:** GeoTransolver / Transolver on **volume** need `model.inference_domain: volume` and stats that match volume training (`global_stats.json` with `velocity`, `pressure_volume`, `turbulent_viscosity`, or `volume_fields_normalization.npz`). **Surface** GeoTransolver/Transolver needs surface stats (`pressure` + `shear_stress` in `global_stats.json` or `surface_fields_normalization.npz`). DoMINO uses `domino_config` for volume.

**Custom code:** [Custom models](#custom-models-adding-a-new-wrapper) · [Custom datasets](#custom-datasets-adding-a-new-adapter) · [Baseline stubs](#baseline-model-stubs-surface_baseline-volume_baseline)

---

## Metrics

Registered names (or dicts with `name` + kwargs). Example list from `evaluation_config.yaml`:

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

The benchmark engine writes **`benchmark_results.json` / `.csv` / `.html`**, then optional **`reports.visuals`** PNGs (when `reports.enabled` and `visuals` is non-empty). Comparison VTK on disk exists only if **`reports.save_comparison_meshes: true`** (otherwise PNGs use in-memory meshes). **`run.save_inference_mesh: true`** adds prediction-only VTK per case.

Built-in **visual** names:

| Name | Role |
| ---- | ---- |
| `field_comparison_surface` | Surface GT vs pred (`plot_field_comparisons`) |
| `plot_fields_volume` | Volume scalars (`plot_fields`) |
| `line_plot` / `plot_line` | GT vs pred along `plot_coord` on comparison mesh |
| `design_scatter` | Pred vs true scatter + R²; needs **`pairs`**: `[{name, true_key, pred_key}]` matching `per_case[].metrics` |
| `design_trend` | Trend vs index; same **`pairs`**, optional **`idx_key`** per pair for x labels |
| `projections_hexbin` | Hexbin over many meshes; needs **`mesh_paths`**, **`field`**, **`direction`** (see `bench.visualization.plot_projections_hexbin`) |
| `streamlines_comparison` | Volume streamlines (point comparison mesh); uses **`output.streamlines_vector_canonical`** or **`canonical_key`** |

Register more: `physicsnemo.cfd.evaluation.reports.register_visual`.

**Runtime context:** `run_optional_report_plugins(..., context={"comparison_meshes_by_run": [{case_id: mesh}, ...]})` aligns with `results` so built-ins can skip `pv.read` for large VTU/VTP.

**Outputs:** PNGs under `{run.output_dir}/visuals/`; manifest `report_plugins_manifest.json`; optional comparison mesh `{run.output_dir}/{comparison_mesh_subdir}/{case_id}_comparison.vtp|vtu`.

**Inference meshes:** (1) `inference_<model>_<case>.vtp|vtu` — **predictions only** (optional via `run.save_inference_mesh`); (2) optional **`comparison_meshes/<case>_comparison.*`** — **GT + pred** when `save_comparison_meshes` is true.

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
    - name: line_plot
      case_ids: ["run_1"]
      canonical_key: pressure
      plot_coord: x
      normalize_factor: 1.0
```

| Key | Role |
| --- | ---- |
| `enabled` | If true, run registered visuals after metrics/tables. |
| `save_comparison_meshes` | Save comparison VTK to disk. If false, mesh-based visuals still run from memory when the runner built a comparison mesh. |
| `comparison_mesh_subdir` | Subfolder for `*_comparison.vtp|vtu`. |
| `visual_case_ids` | Optional. If set, only these case IDs are kept in RAM for mesh-backed PNGs; also becomes the default `case_ids` for any visual that omits it (per-visual `case_ids` overrides). Omit for legacy behavior (all cases in memory). |
| `visuals` | Ordered list: string names or `{ name: ..., ...kwargs }`. |
| `plugins` | Legacy manifest hooks (`ReportsConfig`). |

### Field selection (built-ins)

- **`field_comparison_surface`** — Uses `canonical_keys` (default `pressure`) and `output.ground_truth_mesh_field_names` / `mesh_field_names`.
- **`plot_fields_volume`** — `fields` = VTK array names, or default all `output.volume_mesh_field_names` values.
- **`line_plot`** — One GT vs pred line per case; **`canonical_key`** picks surface or volume maps; optional **`coord_trim`** / **`field_trim`** (two-element lists), **`flip`**, matplotlib kwargs forwarded to `plot_line` (e.g. `xlabel`, `ylabel`, `true_line_kwargs`).
- **`design_scatter` / `design_trend`** — Require **`pairs`** in YAML; values must appear under **`per_case[].metrics`** (e.g. from custom metrics or extended metric dicts).
- **`projections_hexbin`** — **`mesh_paths`**: explicit list of VTK paths (same workflow as [`workflows/bench_example`](../bench_example/) projection outputs).
- **`streamlines_comparison`** — Volume cases only (`metric_dtype` point); uses GT/pred VTK names from **`output`** for the chosen canonical key.

### Bench example ↔ evaluation mapping

| `workflows/bench_example` / `bench.visualization.utils` | Evaluation visual name |
| --------------------------------------------------------- | ------------------------ |
| `plot_field_comparisons` (contours / utils) | `field_comparison_surface` |
| `plot_fields` | `plot_fields_volume` |
| `plot_line` | `line_plot` |
| `plot_design_scatter` | `design_scatter` |
| `plot_design_trend` | `design_trend` |
| `plot_projections_hexbin` | `projections_hexbin` (`mesh_paths` in YAML) |
| `plot_streamlines` | `streamlines_comparison` |
| `get_visible_point_indices` | (no built-in; used inside bench_example contour scripts only) |

### Line plots

Built-in **`line_plot`** wraps **`plot_line`**: it loads the comparison mesh, builds a **cell-center** point cloud (or uses points if there are no cells), and plots GT vs pred vs sorted **`plot_coord`** (`x` \| `y` \| `z`). Set **`canonical_key`** to a surface key (`pressure`, `shear_stress`) or volume key (`pressure_volume`, …) present in **`output`**. For **centerline-style** plots like [`workflows/bench_example`](../bench_example/README.md), use those scripts or a **custom** `register_visual` that passes a list of polylines to `plot_line`.

### Other plot paths

- **Bench example:** run inference to get VTK, then `generate_surface_benchmarks.py` / `generate_volume_benchmarks.py` with a matching field map.
- **Programmatic:** `bench.visualization.utils` or `run_optional_report_plugins(config, results, output_dir, context={...})` with `per_case[].comparison_mesh_path` and/or **`context["comparison_meshes_by_run"]`**.

### Troubleshooting

- **`case_ids`** in a visual must include the case you run — use `--case-id run_1` if the visual lists `case_ids: ["run_1"]`, or omit `case_ids`.
- Use an **editable install** or **`PYTHONPATH`** to this repo so the benchmark run includes the visualization step (not an old wheel).
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

