# Evaluation examples (PhysicsNeMo-CFD)

Example YAML and scripts for **`physicsnemo.cfd.evaluation`**: config-driven inference and benchmarking (same stack as the `nvidia-physicsnemo-cfd` package). Install the repo or `pip install nvidia-physicsnemo-cfd`, then run the CLIs below from this directory.

## Inference and benchmarking layer (`physicsnemo.cfd.evaluation`)

Configure model, dataset, metrics, and checkpoint via YAML; run single-case inference or a full benchmark with JSON/CSV/HTML reports. Metrics delegate to **`physicsnemo.cfd.bench`** when a comparison mesh is built (see `output.mesh_field_names` and ground-truth maps in config).

### Quick start

**Single-case inference** (one model, one case; writes prediction mesh):

```bash
python -m physicsnemo.cfd.evaluation.inference.run --config inference_config.yaml
python -m physicsnemo.cfd.evaluation.inference.run --config inference_config.yaml --case-id run_1
```

**Shorthand** (same entrypoints):

```bash
python -m physicsnemo.cfd.evaluation.inference
```

**Benchmark** (model × dataset, metrics, JSON/CSV/HTML report) — merge shared defaults + matrix overlay:

```bash
python -m physicsnemo.cfd.evaluation.benchmarks.run --base-config inference_config.yaml --config benchmark_matrix.yaml --case-id run_1
```

```bash
python -m physicsnemo.cfd.evaluation.benchmarks
```

You can also use a **single** full YAML (no `--base-config`) if everything lives in one file.

**CLI overrides** (without editing YAML):

```bash
python -m physicsnemo.cfd.evaluation.benchmarks.run --base-config inference_config.yaml --config benchmark_matrix.yaml run.device=cpu run.output_dir=./out
# or with -- prefix:
python -m physicsnemo.cfd.evaluation.benchmarks.run --base-config inference_config.yaml --config benchmark_matrix.yaml --run.device=cuda:1 --model.checkpoint=path/to/ckpt.pth
```

Optional **`--base-config`** works for `physicsnemo.cfd.evaluation.inference.run` too (merge order: base, then `--config`).

**Minimal surface-only example** (placeholders; edit paths): `surface_benchmark_minimal.yaml`.

### Config

Edit `inference_config.yaml` (shared run/model/dataset/output/metrics and default `benchmark.mode: single`) and, for matrix sweeps, `benchmark_matrix.yaml` (typically only a `benchmark:` block). JSON works as well.

- **run**: `device`, `output_dir`, `seed`, `batch_size`
- **model**: `name` (registered wrapper id: fignet, xmgn, geotransolver, … or your [custom model](#custom-models-adding-a-new-wrapper)), `checkpoint`, `stats_path`, optional **`inference_domain`** (`surface` \| `volume`) to override the wrapper default (required for **GeoTransolver** / **Transolver** on VTU volume cases: use `volume` with a volume-trained checkpoint and **`global_stats.json`** (preferred; must include `velocity`, `pressure_volume`, `turbulent_viscosity`) or `volume_fields_normalization.npz` beside the checkpoint, as in `transformer_models/src/inference_on_vtk.py`; **DoMINO** uses the same key with a volume `model.model_type` in `domino_config` and VTU cases), `kwargs` (e.g. `max_points`, `node_degree`, `interpolation_k`; GeoTransolver/Transolver: `resolution`, `geometry_sampling`, `air_density`, `stream_velocity`, `batch_resolution`; optional datapipe overrides: `include_sdf`, `translational_invariance`, `scale_invariance`, `reference_scale`, `broadcast_global_features`, …; **DoMINO**: `domino_config`, `point_batch_size`). The **baseline** names are optional no-checkpoint stubs; see [Baseline model stubs](#baseline-model-stubs-surface_baseline-volume_baseline) below.
- **dataset**: `name` (registered adapter id: drivaerml, ahmed, or your [custom dataset](#custom-datasets-adding-a-new-adapter)), `root`, `split`, `case_ids` (null = all cases), optional `kwargs` e.g. `align_ground_truth_to_model: true` or `gt_data_type: from_model` so reference fields are read at the same mesh location as the model (point vs cell) without separate dataset entries per model in matrix benchmarks. **DrivAerML volume**: same `root` and `run_*` dirs as surface; set `inference_domain: volume` and place VTU files per run (default name `volume_<n>.vtu` for `run_<n>`, e.g. `run_1/volume_1.vtu`). Optional `volume_vtu_filename` or `volume_vtu_template` (e.g. `"volume_{run_suffix}.vtu"`) if your naming differs.
- **output** (optional): `mesh_field_names` / **`ground_truth_mesh_field_names`** — canonical surface keys → VTP array names for predictions vs reference. **`volume_mesh_field_names`** / **`ground_truth_volume_mesh_field_names`** — volume keys → VTU names; includes **`pressure_volume`** (volume pressure, distinct from surface `pressure`), plus `turbulent_viscosity`, `velocity` (defaults often `pMeanPred`, `nutMeanPred`, `UMeanPred` for preds, `pMean`, `nutMean`, `UMean` for GT).
- **metrics**: list of names or dicts with `name` + kwargs, e.g. `l2_pressure`, `l2_shear_stress`, **`l2_pressure_volume`** (volume pressure / `pressure_volume`), `l2_velocity`, **`l2_turbulent_viscosity`** (optional `gt_key` / `pred_key`), `drag_error`, `lift_error`, `continuity_residual_l2`, `momentum_residual_l2`. Metric kwargs are passed through from the config entry.
- **reports** (optional): `enabled`, `plugins` — post-scalar manifest hooks (see `physicsnemo.cfd.evaluation.config.ReportsConfig`).
- **benchmark**: `mode` (single | matrix), `models` / `datasets` for matrix mode, `reproducibility`. In **matrix** mode, pairs where the model **`INFERENCE_DOMAIN`** does not match the dataset adapter domain (from `dataset.kwargs`, e.g. DrivAerML `inference_domain: volume`) are **skipped**: a line is printed, rows appear in CSV/HTML as `_skipped`, and `benchmark_artifacts.json` lists `skipped_runs`.

### Baseline model stubs (surface_baseline, volume_baseline)

They are **optional** registered models—not required for normal inference with trained checkpoints.

**Purpose**

- **Smoke tests & CI** — Exercise the full path (dataset adapter → `CanonicalCase` → `CFDModel.load` / `prepare_inputs` / `predict` / `decode_outputs` → metrics and, for inference, VTK write) **without** loading a neural network, **without** valid `checkpoint` / `stats_path`, and **without** a GPU-heavy forward pass.
- **Layout & domain checks** — Confirm case listing, mesh paths (boundary VTP vs volume VTU), `inference_domain` matching, and benchmark matrix **skip** behavior for incompatible model×dataset pairs.
- **Debugging** — Isolate data I/O and reporting when a real model fails early (bad stats, OOM, etc.).

**What they are *not***

- They are **not** chosen automatically from a path like `*_volume_checkpoint`; only **`model.name: surface_baseline`** or **`volume_baseline`** selects them.
- They are **not** physics-accurate predictions: outputs are **zeros** (or shape-only placeholders) with the right **canonical keys** for the domain (`pressure` / `shear_stress` on surface, `pressure_volume` / `turbulent_viscosity` / `velocity` on volume).
- **`surface_baseline`** uses **cell-centered** counts on the boundary mesh; if your ground truth is **point**-aligned, align GT via `dataset.kwargs` (e.g. `align_ground_truth_to_model` / `gt_data_type`) or expect metric shape mismatches or NaNs.

**Config sketch**

```yaml
model:
  name: "surface_baseline"   # or "volume_baseline"
  checkpoint: ""
  stats_path: ""
dataset:
  name: drivaerml
  root: /path/to/data
  kwargs: {}                 # surface default
  # kwargs: { inference_domain: volume }   # required for volume_baseline
```

If you never use smoke tests or matrix stubs, you can ignore these names; the rest of the stack does not depend on them.

### Custom models (adding a new wrapper)

Built-in models (fignet, xmgn, geotransolver, …) are **registered** `CFDModel` implementations. To add your own architecture or checkpoint layout, implement a **wrapper** and register it once; inference and benchmarks pick it up by `model.name` in YAML.

#### Where to put code

- Add a Python module under **`physicsnemo/cfd/evaluation/inference/wrappers/`**, e.g. `wrappers/my_model/wrapper.py` or `wrappers/my_model.py`.
- Keep imports under `physicsnemo.cfd.evaluation` (and `nvidia-physicsnemo` / PyTorch as needed). Do **not** add inference code under `physicsnemo/cfd/inference/` — that tree is reserved for the DoMINO NIM client.

#### Implement `CFDModel`

Subclass **`CFDModel`** from `physicsnemo.cfd.evaluation.inference.model_registry` and implement:

| Piece | Purpose |
| ----- | --- |
| **`INFERENCE_DOMAIN`** | Class variable: `"surface"` (boundary VTP) or `"volume"` (volume VTU). Must match the dataset adapter domain for benchmark runs. |
| **`OUTPUT_LOCATION`** + **`output_location`** | `"point"` or `"cell"` — where your predictions live on the mesh (drives GT alignment and writing). |
| **`load(checkpoint_path, stats_path, device, **kwargs)`** | Load weights and stats; **`kwargs`** includes `model.kwargs` from YAML and `inference_domain` when set. Return `self`. |
| **`prepare_inputs(case: CanonicalCase)`** | Build tensors / graphs / batches from `case.mesh_path` and optional `ground_truth` / `metadata`. |
| **`predict(model_input)`** | One forward pass; return raw outputs (tensors, dict, etc.). |
| **`decode_outputs(raw_output, case)`** | Map to the **canonical predictions dict** used by metrics and the mesh bridge: **surface** — `pressure`, `shear_stress`; **volume** — `pressure_volume`, `turbulent_viscosity`, `velocity` (use `build_predictions_dict` / `predictions_dict` from `physicsnemo.cfd.evaluation.datasets.schema`). |

If your wrapper’s class defaults to `INFERENCE_DOMAIN = "surface"` but you train a VTU volume checkpoint, users set **`model.inference_domain: volume`** in YAML so the engine matches volume datasets.

#### Register the wrapper

1. In **`physicsnemo/cfd/evaluation/inference/wrappers/__init__.py`**, import your class and call **`register_model("my_model", MyModelWrapper)`** so the module is loaded at startup (same pattern as existing wrappers).
2. Reinstall or run with the repo on **`PYTHONPATH`** / `pip install -e .` so the new module is importable.

#### Use in YAML

```yaml
model:
  name: my_model
  checkpoint: /path/to/weights.pt
  stats_path: /path/to/dir_with_stats   # or file, per your load()
  inference_domain: surface             # optional override if your class default differs
  kwargs:
    batch_size: 1
    # any options your load() / prepare_inputs() read
```

No changes to **`run.py`** or **`benchmarks/engine.py`** are required.

#### Reference implementations (in-repo)

| Wrapper | Notes |
| ------- | --- |
| `wrappers/fignet/wrapper.py` | Cell-centered surface |
| `wrappers/xmgn/wrapper.py` | Point-based surface |
| `wrappers/transolver/wrapper.py`, `wrappers/geotransolver/wrapper.py` | Surface, cell; boundary VTP + STL + `surface_fields_normalization.npz` |
| `wrappers/domino/wrapper.py` | DoMINO config + scaling |
| `wrappers/surface_baseline.py`, `wrappers/volume_baseline.py` | No checkpoint; smoke tests |

### Custom datasets (adding a new adapter)

Built-in datasets (drivaerml, ahmed, …) are **registered** `DatasetAdapter` implementations. To plug in your own layout (folder naming, file formats, splits), add an **adapter** and register it once; inference and benchmarks resolve it by `dataset.name` in YAML.

#### Where to put code

- Add a module under **`physicsnemo/cfd/evaluation/datasets/adapters/`**, e.g. `my_dataset.py`.
- Shared helpers (VTK GT extraction, alignment) live under **`physicsnemo/cfd/evaluation/datasets/`** (`vtk_ground_truth.py`, `gt_alignment.py`, …)—reuse them when possible instead of duplicating I/O.

#### Implement `DatasetAdapter`

Subclass **`DatasetAdapter`** from `physicsnemo.cfd.evaluation.datasets.adapter_registry` (also exported as `physicsnemo.cfd.evaluation.datasets.DatasetAdapter`) and implement:

| Piece | Purpose |
| ----- | --- |
| **`__init__(self, root: str, **kwargs)`** | Store **`root`** (required dataset root) and options from **`dataset.kwargs`** in YAML. Validate paths early if you need to fail fast. |
| **`list_cases(split=None)`** | Return case IDs (strings). **`split`** is adapter-defined (`"train"` / `"test"` / `None` = all); document what your dataset supports. |
| **`load_case(case_id: str)`** | Return a **`CanonicalCase`**: `case_id`, **`mesh_path`** (absolute or root-relative path to the primary **`.vtp`** or **`.vtu`**), **`mesh_type`** (`"point"` \| `"cell"` — how GT fields in `ground_truth` align to the mesh), optional **`ground_truth`** (numpy arrays keyed for surface: `pressure`, `shear_stress`; volume: `pressure_volume`, `velocity`, `turbulent_viscosity`, …), optional **`metadata`**, and **`inference_domain`** (`"surface"` \| `"volume"`). |
| **`inference_domain_from_kwargs`** (classmethod, optional) | If domain depends on kwargs (e.g. DrivAerML surface vs volume), return `"surface"` or `"volume"` so matrix benchmarks can **skip** incompatible model×dataset pairs **before** loading cases. |

Ground truth in **`ground_truth`** should match what your metrics and **`mesh_bridge`** expect: canonical keys, with shapes consistent with **`mesh_type`** and the model’s **`OUTPUT_LOCATION`** (use **`dataset.kwargs`** like `align_ground_truth_to_model` / `gt_data_type` where supported—see `datasets/gt_alignment.py`).

#### Register the adapter

1. In **`physicsnemo/cfd/evaluation/datasets/adapters/__init__.py`**, import your class and call **`register_adapter("my_dataset", MyDatasetAdapter)`** so it loads with `import physicsnemo.cfd.evaluation.datasets.adapters`.
2. Reinstall or use **`pip install -e .`** / **`PYTHONPATH`** to the repo root.

#### Use in YAML

```yaml
dataset:
  name: my_dataset
  root: /path/to/dataset_root
  split: null
  case_ids: null
  kwargs:
    # adapter-specific: splits, filename templates, inference_domain, GT field names, etc.
    inference_domain: surface
```

The engine passes **`dataset.kwargs`** into your adapter’s constructor (merged with alignment helpers for known keys—see DrivAerML).

#### Canonical types (`physicsnemo.cfd.evaluation.datasets.schema`)

- **`CanonicalCase`**: `case_id`, `mesh_path`, `mesh_type`, `ground_truth`, `metadata`, `inference_domain`. Produced by adapters; consumed by model wrappers and metrics.
- **Predictions** (model output, not dataset): surface — `pressure`, `shear_stress`; volume — **`pressure_volume`**, `turbulent_viscosity`, `velocity`. Helpers: `predictions_dict`, `build_predictions_dict`.

#### Reference implementations (in-repo)

| Adapter | Notes |
| ------- | --- |
| `adapters/drivaerml.py` | **Surface**: `run_<n>/boundary_<n>.vtp`, GT `pressure` / `shear_stress`; optional kwargs: `boundary_vtp_filename`, `boundary_vtp_template`, field name overrides. **Volume**: `kwargs.inference_domain: volume`, `run_<n>/volume_<n>.vtu`, GT `pressure_volume`, `turbulent_viscosity`, `velocity` via `extract_volume_fields_from_mesh`; optional `volume_vtu_filename` / `volume_vtu_template`. |
| `adapters/ahmed.py` | Stub for layout / registration tests |

See **`datasets/vtk_ground_truth.py`**, **`datasets/gt_alignment.py`** for resampling and naming. For **point** vs **cell** GT alignment with MeshGraphNet-style models, use kwargs such as **`align_ground_truth_to_model`** / **`gt_data_type`** as in DrivAerML.

### Package layout (in this repo)

Paths are relative to the repository root:

- `physicsnemo/cfd/evaluation/config.py` — config schema, loader, and `deep_merge_dict` / `Config.load_merged` for multi-file configs
- `inference_config.yaml` / `benchmark_matrix.yaml` / `surface_benchmark_minimal.yaml` — examples in this folder (`--base-config` + `--config` where applicable)
- `physicsnemo/cfd/evaluation/inference/` — model registry; `common_wrapper_utils/vtk_datapipe_io.py`; `wrappers/<model>/`
- `physicsnemo/cfd/evaluation/datasets/` — canonical case schema and adapters (drivaerml, ahmed stub)
- `physicsnemo/cfd/evaluation/metrics/` — registry + `mesh_bridge.py` + `builtin/` (thin wrappers around `physicsnemo.cfd.bench`)
- `physicsnemo/cfd/evaluation/benchmarks/` — engine, reporting (JSON, CSV, HTML), optional `report_plugins.py`
- `physicsnemo/cfd/evaluation/common/` — mesh I/O, stats, kNN interpolation
- `physicsnemo/cfd/bench/` — shared algorithms (L2, forces, physics, …); not duplicated under `evaluation/`

### Migration from external `physicsnemo_cfd`

If you used the old standalone package name **`physicsnemo_cfd`**, switch imports to **`physicsnemo.cfd.evaluation`** and run:

`python -m physicsnemo.cfd.evaluation.benchmarks.run --config ...`  
(or `python -m physicsnemo.cfd.evaluation.benchmarks`).
