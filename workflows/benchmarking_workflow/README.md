# Model evaluation and benchmarking

This workflow runs **metrics**, tabular outputs (JSON/CSV/HTML), optional **PNG visuals**, and optional VTK using **[Hydra](https://hydra.cc/)** and **OmegaConf** — the same pattern as **`workflows/domino_design_sensitivities/`**. It ties together model inference, built-in metrics, and artifacts for analysis.

**Contributing:** see the repository **[CONTRIBUTING.md](../../CONTRIBUTING.md)** for pull requests, tests, and sign-off requirements.

---

## Installation

Install **PhysicsNeMo-CFD** from the repository root (setuptools reads **[`pyproject.toml`](../../pyproject.toml)**; there is no separate `setup.py`):

```bash
cd physicsnemo-cfd   # repository root
pip install -e ".[dev]"   # editable install + pytest for contributors
# optional GPU extras:
# pip install -e ".[gpu]" --extra-index-url=https://pypi.nvidia.com
```

Then use the commands below **from this directory** (`workflows/benchmarking_workflow/`).

---

## Quick start

1. **Prerequisites:** Python 3.10+, NVIDIA GPU for full inference runs, checkpoints and dataset on disk or mounted volume.
2. **Assets:** Download benchmark checkpoints and the **DrivAerML** evaluation tree (see [DrivAerML dataset: download and directory layout](#drivaerml-dataset-download-and-directory-layout)); **NGC** may mirror these later (see [Benchmark assets on NGC](#benchmark-assets-on-ngc-coming-soon)). Point `model.checkpoint`, `stats_path`, and `dataset.root` in `conf/*.yaml` at local paths, or use [Hydra overrides](#paths-without-ngc-hydra-overrides).
3. **Configure:** Edit `conf/config_surface.yaml` (default), `config_volume.yaml`, or matrix configs under `conf/`.
4. **Run:**

```bash
# Surface benchmark (default config)
python main.py

# Volume benchmark
python main.py --config-name=config_volume

# Matrix: multiple models × dataset blocks
python main.py --config-name=config_matrix_surface
python main.py --config-name=config_matrix_volume

# Overrides (examples)
python main.py case_id=run_1 run.device=cuda:0
python main.py --config-name=config_volume run.output_dir=my_volume_run
python main.py run.fail_on_all_skipped=true
```

5. **Outputs:** Under `run.output_dir` — `benchmark_results.json` / `.csv` / `.html`, optional `benchmark_artifacts.json`, `metrics_cache/` when enabled, Hydra metadata under `hydra/` if configured.
6. **Multi-GPU (optional):** install **physicsnemo** so `DistributedManager` is available, then:

```bash
torchrun --standalone --nproc_per_node=4 main.py
# or: torchrun --standalone --nproc_per_node=8 main.py --config-name=config_matrix_surface
```

Cases are split across GPUs (`cases[rank::world_size]`). Rank 0 writes reports and optional artifacts. Set `run.distributed: false` only for debugging (each rank would run the full case list).

---

## Benchmark assets on NGC (coming soon)

Pretrained checkpoints and the ground-truth evaluation dataset are intended to be published on **NVIDIA NGC**. **Links are not yet public** — use local paths in config until then.

| Resource | Description | NGC link |
| -------- | ----------- | -------- |
| Benchmark model checkpoints | Surface/volume checkpoints + `global_stats.json` (or equivalent) per model | *TBD* |
| Evaluation dataset | DrivAerML (or successor) — same **`run_*`** layout as [DrivAerML dataset](#drivaerml-dataset-download-and-directory-layout) | *TBD* |

After release, unpack assets to a stable path and set `benchmark.models[].checkpoint` / `stats_path` and `benchmark.datasets[].root` to match the documented layout (or use env-driven paths in your deployment). Until then, use Hugging Face or a local mirror as described in that section.

---

## Paths without NGC (Hydra overrides)

Until NGC artifacts are available, override paths from the CLI without editing YAML files:

```bash
python main.py \
  benchmark.models.0.checkpoint=/path/to/checkpoint.pt \
  benchmark.models.0.stats_path=/path/to/global_stats.json \
  benchmark.datasets.0.root=/path/to/drivaer_data
```

You can also export environment variables and reference them in a local, gitignored config using OmegaConf `oc.env` patterns if you add them to your YAML.

---

## DrivAerML dataset: download and directory layout

The **`drivaerml`** adapter expects a **local root directory** (`benchmark.datasets[].root`) whose children are **`run_<id>`** folders. Each run holds the VTK ground-truth meshes the metrics read from.

**Where to get the data**

- **Project / landing page:** [DrivAerML on CAE-ML Datasets](https://caemldatasets.org/drivaerml/) — licensing, citation, and overview.
- **Hugging Face (common for downloads):** [`neashton/drivaerml`](https://huggingface.co/datasets/neashton/drivaerml) — dataset repository you can sync to disk.
- **NGC:** when the [benchmark table](#benchmark-assets-on-ngc-coming-soon) lists a DrivAerML (or successor) bundle, use the documented unpack path the same way as a manually downloaded tree.

The full public release is **large** (many `run_*` directories and VTK files). Plan disk space and use partial sync if you only need a few cases for smoke tests.

**On-disk layout (what the adapter checks)**

| Mode | Per-run directory | Default file |
| ---- | ----------------- | ------------ |
| **Surface** (default) | `root/run_<n>/` | `boundary_<n>.vtp` (e.g. `run_1/boundary_1.vtp`) |
| **Volume** (`dataset.kwargs.inference_domain: volume`) | `root/run_<n>/` | `volume_<n>.vtu` |

Only directories that contain the required mesh for the selected mode are listed as cases. Overrides for non-standard filenames are documented on **`DrivAerMLAdapter`** in **`physicsnemo.cfd.evaluation.datasets.adapters.drivaerml`**.

**Download with the Hugging Face CLI (typical)**

Install the CLI, then download the dataset repo into a directory you will point at as **`dataset.root`**:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download neashton/drivaerml --repo-type dataset --local-dir /path/to/drivaerml_data
```

Set in YAML or via override, for example:

```bash
python main.py benchmark.datasets.0.root=/path/to/drivaerml_data
```

**Alternative: `git` + Git LFS**

If you prefer a Git workflow, clone the Hugging Face dataset repository with **Git LFS** installed so large VTK objects are fetched. The resulting clone root (the folder that contains the `run_*` directories) is the same value for **`benchmark.datasets[].root`**.

**Single-run or scripted downloads**

For a minimal sanity check, you can copy individual `run_<id>` trees (surface `boundary_*.vtp` and any co-located files your workflow needs) under one parent directory so that parent is **`root`**. The notebook [`notebooks/surface_benchmarking.ipynb`](notebooks/surface_benchmarking.ipynb) shows example URLs for fetching files from the Hugging Face dataset for one sample.

**Train / validation split for benchmarking**

After the tree is on disk, use the fixed 90/10 split files under [`drivaer_ml_files/`](drivaer_ml_files/) if you want to align case lists with that split — see [`drivaer_ml_files/README.md`](drivaer_ml_files/README.md).

---

## Notebooks (exploration, not batch production)

The **`notebooks/`** directory (`surface_benchmarking.ipynb`, `volume_benchmarking.ipynb`, etc.) is for **interactive analysis**, plots, and exploring metrics. For **repeatable, automatable, config-driven** benchmarking (including CI-style runs), use **`main.py`** with **`conf/`** so every run is logged and reproducible.

---

## Exit codes (automation)

- By default, **`main.py`** exits **0** when `run_benchmark` completes without raising.
- Set **`run.fail_on_all_skipped: true`** to exit **1** if every model×dataset run was skipped (e.g. domain mismatch in matrix mode).
- Set **`run.fail_on_any_metric_nan: true`** to exit **1** if any aggregate metric in `metrics` is NaN.
- Override on the CLI: `run.fail_on_all_skipped=true`.

The flat CLI **`python -m physicsnemo.cfd.evaluation.benchmarks.run`** applies the same policy.

---

## Config files

| File | Role |
| ---- | ----- |
| [`conf/config_surface.yaml`](conf/config_surface.yaml) | **Surface**, `benchmark.mode: matrix`, **GeoTransolver** × **drivaerml**. Top-level **`case_id: [run_1, run_11]`**. **`run.device`**: `cuda:1`. **`run.output_dir`**: **`benchmark_results_surface`**. |
| [`conf/config_volume.yaml`](conf/config_volume.yaml) | **Volume**, same pattern; **`run.output_dir`**: **`benchmark_results_volume`**. |
| [`conf/config_matrix_surface.yaml`](conf/config_matrix_surface.yaml) | **Matrix surface:** multiple models × one drivaerml row; **`case_id: null`**. **`run.output_dir`**: **`benchmark_results_matrix_surface`**. |
| [`conf/config_matrix_volume.yaml`](conf/config_matrix_volume.yaml) | **Matrix volume:** same for volume checkpoints. **`run.output_dir`**: **`benchmark_results_matrix_volume`**. |

**Matrix mode:** every **`benchmark.models`** entry runs against every **`benchmark.datasets`** entry; incompatible surface/volume pairs are skipped. Edit checkpoint paths; comment out any **`- name:`** block you do not need.

---

## Config reference (YAML keys)

| Section | Contents |
| ------- | -------- |
| **run** | `device`, `output_dir`, `seed`, `batch_size`, **`save_inference_mesh`**, **`distributed`**, **`fail_on_all_skipped`**, **`fail_on_any_metric_nan`**, optional **`metrics_cache`** |
| **model** / **benchmark.models** | `name`, `checkpoint`, `stats_path`, optional **`package`** (`hf://org/repo@rev`, `s3://…`, or local dir), **`checkpoint_relpath`** / **`stats_relpath`** (paths inside the package), `inference_domain`, `kwargs` |
| **dataset** / **benchmark.datasets** | `name`, `root`, `split`, optional `case_ids` (`null` = all), `kwargs` |
| **output** | VTK field name maps; **`streamlines_vector_canonical`** (volume); optional **`surface_interpolate_point_to_cell_for_metrics`** (kNN-IDW point → cell for XmGN/FiGNet-style surfaces so drag/lift/L2 use **`metric_dtype: cell`**), plus **`surface_metrics_idw_k`**, **`surface_metrics_idw_device`** |
| **metrics** | Metric names or `{ name: ..., ...kwargs }` |
| **reports** | Optional PNG pipeline |
| **benchmark** | `mode`, `models` / `datasets`, **`reproducibility`** (`log_env`, `save_artifacts`) |

**`case_id`:** optional top-level Hydra key: `null` uses each dataset's `case_ids` (or all adapter cases); a **string** runs one case on every dataset; a **list** runs those cases in matrix mode.

**`benchmark.reproducibility.log_env`:** when `true`, writes **full `os.environ`** to `env.json` under `run.output_dir` — avoid in shared CI or when secrets may be present. Default in code is `false`; example configs may set `true` for local debugging.

**Metrics cache:** `run.metrics_cache.enabled` stores per-case scalars; delete the cache directory for a full recompute. Plots and meshes are not cached.

**Remote model assets:** Install optional **`pip install 'nvidia-physicsnemo-cfd[evaluation-hf]'`** for `hf://` and `s3://` package roots. Cache directory defaults to `~/.cache/physicsnemo-cfd/models` or override with **`PHYSICSNEMO_CFD_MODEL_CACHE`**. Built-in matrix models (including **domino**, which also resolves **`domino_config`** from the package) use per-model roots in **`physicsnemo.cfd.evaluation.assets.builtin_packages`** when **`checkpoint`** / **`stats_path`** are omitted; override with explicit paths or **`model.package`** as needed. **`register_default_asset`** remains available for custom names. See **[CONTRIBUTING.md](../../CONTRIBUTING.md)** for custom-wrapper tiers.

---

## Custom models, datasets, and metrics

### Custom models

1. Subclass **`CFDModel`** (`physicsnemo.cfd.evaluation.models.model_registry`) under `physicsnemo/cfd/evaluation/models/wrappers/`.
2. Implement `INFERENCE_DOMAIN`, `OUTPUT_LOCATION`, `load`, `prepare_inputs`, `predict`, `decode_outputs`.
3. **`register_model("my_model", MyWrapper)`** in `wrappers/__init__.py`.

### Custom datasets

1. Subclass **`DatasetAdapter`** under `physicsnemo/cfd/evaluation/datasets/adapters/`.
2. Implement `list_cases`, `load_case` → **`CanonicalCase`**.
3. **`register_adapter("my_dataset", MyAdapter)`** in `adapters/__init__.py`.

### Custom metrics

Register functions with **`physicsnemo.cfd.evaluation.metrics.register_metric`**, then list the name under **`metrics:`** in YAML (see **`physicsnemo.cfd.evaluation.metrics`**).

### Canonical types

Schema: **`physicsnemo.cfd.evaluation.datasets.schema`** — **`CanonicalCase`**, prediction keys for surface/volume.

---

## Metrics

Registered names (or dicts with `name` + kwargs). Examples match **`conf/config_surface.yaml`** / **`config_volume.yaml`**:

```yaml
metrics:
  - l2_pressure
  - l2_shear_stress
  - drag
  - lift
```

| Name | Meaning |
| ---- | ------- |
| `l2_pressure`, `l2_shear_stress` | Surface L2 |
| `l2_pressure_area_weighted` | Area-weighted L2 pressure |
| `drag`, `lift` | Coefficient errors (expands to `drag_error`, etc.) |
| `continuity_residual_l2`, `momentum_residual_l2` | Volume residuals (volume configs) |

---

## Reports and plots

When **`reports.enabled`** and **`reports.visuals`** are set, PNGs are written under `{run.output_dir}/visuals/`. Comparison VTK exists if **`reports.save_comparison_meshes: true`**. Headless servers may need **xvfb** (see **`setup.sh`** for an example `apt` line — optional).

| Name | Role |
| ---- | ---- |
| `field_comparison_surface` | Surface GT vs pred |
| `line_plot` | GT vs pred along `plot_coord` |
| `design_scatter` / `design_trend` | Design-of-experiments style plots |
| `streamlines_comparison` | Volume streamlines |

Register more: **`physicsnemo.cfd.evaluation.reports.register_visual`**.

### Legacy `bench_example` mapping

| `workflows/deprecated/bench_example` / `postprocessing_tools.visualization.utils` | Evaluation visual name |
| --------------------------------------------------------------------------------- | ------------------------ |
| `plot_field_comparisons` | `field_comparison_surface` |
| `plot_line` | `line_plot` |
| `plot_design_scatter` | `design_scatter` |
| `plot_design_trend` | `design_trend` |

**`line_plot`:** centerline-style strip plots — see [`workflows/deprecated/bench_example`](../deprecated/bench_example/README.md) or a custom `register_visual`.

---

## Troubleshooting

- **Missing checkpoints / bad paths:** inference fails or skips; verify overrides and `inference_domain`.
- **DrivAerML `root` wrong:** `dataset.root` must be the directory that **directly contains** `run_*` folders with `boundary_*.vtp` (surface) or `volume_*.vtu` (volume), not a parent of an extra nesting level; see [DrivAerML dataset](#drivaerml-dataset-download-and-directory-layout).
- **CUDA OOM:** reduce batch resolution in `model.kwargs` or use a smaller `case_id` set.
- **Matrix skips:** incompatible model vs dataset domain — check logs for `SKIP` lines.
- **Editable install:** use `pip install -e ".[dev]"` so local `physicsnemo.cfd` changes apply.

---

## Advanced (Python API)

**`physicsnemo.cfd.evaluation.benchmarks.engine.run_benchmark`** and **`Config.from_dict`** are for scripts and tests. **`python -m physicsnemo.cfd.evaluation.benchmarks.run`** and **`python -m physicsnemo.cfd.evaluation.inference`** accept flat YAML/JSON **without** Hydra `${...}` interpolation unless you materialize values first. **`physicsnemo.cfd.evaluation.models`** holds **`CFDModel`** and wrappers; **`evaluation.inference`** adds **`log_inference`** and the compatibility CLI.

---

## Baseline model stubs (`surface_baseline`, `volume_baseline`)

Smoke-test wrappers: **`checkpoint: ""`**, **`stats_path: ""`**. Use in **`benchmark.models`** (matrix) or top-level **`model`** (single).

---

## DrivAerML train/validation split

The [`drivaer_ml_files/`](drivaer_ml_files/) directory lists which **`run_*`** IDs fall in the proposed 90/10 train/validation partition. Download the dataset first ([section above](#drivaerml-dataset-download-and-directory-layout)), then use those lists to constrain **`case_id`** or **`dataset.case_ids`**. Details: [`drivaer_ml_files/README.md`](drivaer_ml_files/README.md).

---

## Package layout (repository)

| Path | Role |
| ---- | ---- |
| `physicsnemo/cfd/evaluation/config.py` | `Config`, `load_config` |
| `physicsnemo/cfd/evaluation/benchmarks/` | Engine, reports, Hydra helpers |
| `physicsnemo/cfd/evaluation/reports/` | `register_visual`, built-in plots |
| `physicsnemo/cfd/postprocessing_tools/` | Shared metrics / visualization |
