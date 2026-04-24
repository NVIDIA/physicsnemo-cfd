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
2. **Assets:** Download benchmark checkpoints and the evaluation dataset when they are published on **NGC** (see [Benchmark assets on NGC](#benchmark-assets-on-ngc-coming-soon)). Until links are available, point `model.checkpoint`, `stats_path`, and `dataset.root` in `conf/*.yaml` at your local paths, or use [Hydra overrides](#paths-without-ngc-hydra-overrides).
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
| Evaluation dataset | DrivAerML (or successor) layout expected by the `drivaerml` adapter | *TBD* |

After release, unpack assets to a stable path and set `benchmark.models[].checkpoint` / `stats_path` and `benchmark.datasets[].root` to match the documented layout (or use env-driven paths in your deployment).

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

**Remote model assets:** Install optional **`pip install 'nvidia-physicsnemo-cfd[evaluation-hf]'`** for `hf://` and `s3://` package roots. Cache directory defaults to `~/.cache/physicsnemo-cfd/models` or override with **`PHYSICSNEMO_CFD_MODEL_CACHE`**. Either set both **`checkpoint`** and **`stats_path`** to local files, or set **`package`** plus **`checkpoint_relpath`** / **`stats_relpath`** (or register defaults via **`register_default_asset`** in code). See **[CONTRIBUTING.md](../../CONTRIBUTING.md)** for custom-wrapper tiers.

---

## Custom models, datasets, and metrics

### Custom models

1. Subclass **`CFDModel`** (`physicsnemo.cfd.evaluation.inference.model_registry`) under `physicsnemo/cfd/evaluation/inference/wrappers/`.
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
- **CUDA OOM:** reduce batch resolution in `model.kwargs` or use a smaller `case_id` set.
- **Matrix skips:** incompatible model vs dataset domain — check logs for `SKIP` lines.
- **Editable install:** use `pip install -e ".[dev]"` so local `physicsnemo.cfd` changes apply.

---

## Advanced (Python API)

**`physicsnemo.cfd.evaluation.benchmarks.engine.run_benchmark`** and **`Config.from_dict`** are for scripts and tests. **`python -m physicsnemo.cfd.evaluation.benchmarks.run`** and **`evaluation.inference`** accept flat YAML/JSON **without** Hydra `${...}` interpolation unless you materialize values first.

---

## Baseline model stubs (`surface_baseline`, `volume_baseline`)

Smoke-test wrappers: **`checkpoint: ""`**, **`stats_path: ""`**. Use in **`benchmark.models`** (matrix) or top-level **`model`** (single).

---

## DrivAerML train/validation split

The [`drivaer_ml_files/`](drivaer_ml_files/) directory contains a reproducible 90/10 train/validation split. See [`drivaer_ml_files/README.md`](drivaer_ml_files/README.md).

---

## Package layout (repository)

| Path | Role |
| ---- | ---- |
| `physicsnemo/cfd/evaluation/config.py` | `Config`, `load_config` |
| `physicsnemo/cfd/evaluation/benchmarks/` | Engine, reports, Hydra helpers |
| `physicsnemo/cfd/evaluation/reports/` | `register_visual`, built-in plots |
| `physicsnemo/cfd/postprocessing_tools/` | Shared metrics / visualization |
