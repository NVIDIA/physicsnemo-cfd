# Model evaluation and benchmarking

This workflow is an **opinionated**, **config-driven** benchmark entry point for **PhysicsNeMo-CFD**: **[Hydra](https://hydra.cc/)** and **OmegaConf** wire model inference, built-in metrics, tabular artifacts (JSON/CSV/HTML), optional **PNG visuals**, and optional VTK. It follows the same pattern as **`workflows/domino_design_sensitivities/`**. Defaults assume **DrivAerML-style** run layouts and evaluation components shipped under **`physicsnemo.cfd.evaluation`**. Nothing here is unique to that stack—you **can extend** wrappers, adapters, metrics, reports, and Hydra compositions as needed (see [**Custom models, datasets, and metrics**](#custom-models-datasets-and-metrics)).

### How to use this README (goal-oriented)

1. **Run the benchmark as configured** — install, set **`benchmark.datasets[].root`** and model paths (or overrides), run **`python main.py`** (default **`config_matrix_surface_hf`**) or **`--config-name=config_matrix_surface_custom`** / **`config_matrix_volume_hf`**, confirm outputs under **`run.output_dir`**.
2. **Stay in YAML / Hydra first** — change YAML under [`conf/`](conf/) (see for example [`config_matrix_volume_hf.yaml`](conf/config_matrix_volume_hf.yaml)) or pass **Hydra overrides** for models, **`case_id`**, metrics, outputs, metrics cache, and exit policy; avoid Python changes until defaults are insufficient.
3. **Extend when needed** — add **`CFDModel`** wrappers, **`DatasetAdapter`** implementations, **`register_metric`** / **`register_visual`**, or custom configs for integrations the stock workflow does not cover.

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

Then use the commands below **from this directory** (`workflows/benchmarking/`).

---

## Quick start

1. **Prerequisites:** Python 3.10+, NVIDIA GPU for full inference runs, checkpoints and dataset on disk or mounted volume. (For a concise **goal-oriented** path through docs and config, see the [introduction](#model-evaluation-and-benchmarking) above.)
2. **Assets:** Download benchmark checkpoints and the **DrivAerML** evaluation tree (see [DrivAerML dataset: download and directory layout](#drivaerml-dataset-download-and-directory-layout)); **NGC** may mirror these later (see [Benchmark assets on NGC](#benchmark-assets-on-ngc-coming-soon)). Point `model.checkpoint`, `stats_path`, and `dataset.root` in `conf/*.yaml` at local paths, or use [Hydra overrides](#paths-without-ngc-hydra-overrides).
3. **Configure:** Edit YAML under `conf/` (see [Configuration files](#configuration-files-in-conf): **`_custom`** = local **`checkpoint`** / **`stats_path`**, **`_hf`** = Hugging Face **`builtin_packages`** when those keys are omitted).
4. **Run:**

```bash
# Matrix surface (default Hydra config: Hugging Face checkpoints)
python main.py

# Local checkpoint paths (edit paths in YAML first)
python main.py --config-name=config_matrix_surface_custom

# Matrix volume — HF vs local
python main.py --config-name=config_matrix_volume_hf
python main.py --config-name=config_matrix_volume_custom

# Overrides (examples)
python main.py case_id=run_1 run.device=cuda:0
python main.py --config-name=config_matrix_surface_hf run.output_dir=my_surface_run
python main.py run.fail_on_all_skipped=true
```

5. **Outputs:** Under `run.output_dir` — `benchmark_results.json` / `.csv` / `.html`, optional `benchmark_artifacts.json`, `metrics_cache/` when enabled, Hydra metadata under `hydra/` if configured.
6. **Multi-GPU (optional):** install **physicsnemo** so `DistributedManager` is available, then:

```bash
torchrun --standalone --nproc_per_node=4 main.py
# or: torchrun --standalone --nproc_per_node=8 main.py --config-name=config_matrix_surface_custom
```

Cases are split across GPUs (`cases[rank::world_size]`). Rank 0 writes reports and optional artifacts. Set `run.distributed: false` only for debugging (each rank would run the full case list).

---

## Configuration files in `conf/`

Hydra loads **`--config-name=<stem>`** from this directory (no `.yaml` suffix). This workflow ships **four** configs: **matrix only** — **`_custom`** for **local or NFS checkpoints** you set in YAML, **`_hf`** for **Hugging Face** defaults via **`builtin_packages`** when **`checkpoint`** / **`stats_path`** are omitted.

| Config file | Role |
| ----------- | ---- |
| **`config_matrix_surface_custom.yaml`** | **Matrix** surface: **`benchmark.models`** × **`benchmark.datasets`**. Fill **`checkpoint`**, **`stats_path`**, **`dataset.root`**, and DoMINO **`kwargs.domino_config`** / **`stats_path` → `scaling_factors.pkl`** as in the template comments. Use **`--config-name=config_matrix_surface_custom`**. |
| **`config_matrix_volume_custom.yaml`** | **Matrix** volume with the same **explicit path** story for VTU inference. |
| **`config_matrix_surface_hf.yaml`** | Same matrix **layout** as the surface **custom** file, but models typically **omit** **`checkpoint`** / **`stats_path`** so weights resolve from **Hugging Face**. This is the **default** for **`python main.py`** (`main.py` `config_name`). Requires **`huggingface-cli login`** or **`HF_TOKEN`** for private repos. |
| **`config_matrix_volume_hf.yaml`** | HF-hosted volume matrix (GeoTransolver / Transolver / DoMINO); **XMGN / FiGNet volume** may have no Hub checkpoint—omit those blocks. |

**`_custom` vs `_hf`:** use **`_custom`** when you want full control over **on-disk** checkpoints and stats; use **`_hf`** when you rely on **Hub** resolution and optional cache under **`PHYSICSNEMO_CFD_MODEL_CACHE`**.

**VTK field names on written meshes:** Under **`output`**, the maps **`mesh_field_names`** / **`ground_truth_mesh_field_names`** (surface) and **`volume_mesh_field_names`** / **`ground_truth_volume_mesh_field_names`** (volume) set the **VTK array names** used when the workflow writes optional **`.vtp`** / **`.vtu`** files (for example saved inference meshes from **`run.save_inference_mesh`** or comparison meshes when reports enable them). Choose strings that match your model’s written fields and the ground-truth adapter’s field names.

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

**Canonical source:** [Hugging Face `neashton/drivaerml`](https://huggingface.co/datasets/neashton/drivaerml) — dataset card, file layout, and download examples (also mirrored in spirit on [CAE-ML Datasets](https://caemldatasets.org/drivaerml/)). License: **CC BY-SA 4.0**; cite the [DrivAerML paper](https://arxiv.org/abs/2408.11969) per the card.

**Scale:** the full repository is on the order of **~31 TB**. Do **not** blindly run a whole-repo sync on a laptop; use **Git LFS** only when you intend to host the full tree, or use **selective** HTTP / CLI downloads for testing.

**What each `run_<i>/` contains (per the dataset card)**

| File / folder | Role |
| ------------- | ---- |
| `drivaer_<i>.stl` | Geometry |
| `geo_ref_<i>.csv`, `geo_parameters_<i>.csv` | Reference / parameter metadata |
| **`boundary_<i>.vtp`** | **Surface** mesh + fields — **required for surface benchmarking** |
| **`volume_<i>.vtu`** | **Volume** mesh — **required for volume benchmarking** (see note below) |
| `force_mom_<i>.csv`, `force_mom_constref_<i>.csv` | Forces / moments |
| `slices/`, `Images/` | Slices and images (optional for this workflow) |

Repo-level extras include `openfoam_meshes/`, `force_mom_all.csv`, etc.; the adapter does not need those for standard metrics.

**Volume files on Hugging Face:** for many runs, **`volume_<i>.vtu` is split** on the Hub into **two parts**. Download both parts for a given `i`, then **concatenate** them into a single `volume_<i>.vtu` in `run_<i>/` before running volume benchmarks (see the [dataset card](https://huggingface.co/datasets/neashton/drivaerml) for naming). Surface-only tests only need **`boundary_<i>.vtp`**.

**On-disk layout (what the adapter checks)**

| Mode | Per-run directory | Default file |
| ---- | ----------------- | ------------ |
| **Surface** (default) | `root/run_<n>/` | `boundary_<n>.vtp` (e.g. `run_1/boundary_1.vtp`) |
| **Volume** (`dataset.kwargs.inference_domain: volume`) | `root/run_<n>/` | `volume_<n>.vtu` |

Only directories that contain the required mesh for the selected mode are listed as cases. Overrides for non-standard filenames are documented on **`DrivAerMLAdapter`** in **`physicsnemo.cfd.evaluation.datasets.adapters.drivaerml`**.

**1) Full mirror — Git + Git LFS (dataset maintainers’ example)**

Install [Git LFS](https://git-lfs.com/), then clone the dataset repo. The clone root (the directory that contains the `run_*` folders) is **`benchmark.datasets[].root`**.

```bash
git lfs install
git clone https://huggingface.co/datasets/neashton/drivaerml
# SSH alternative (if configured on Hugging Face): git clone git@hf.co:datasets/neashton/drivaerml
```

This pulls the **entire** ~31 TB collection when LFS objects are fetched — use only with appropriate storage and bandwidth.

**2) Selective download for testing — HTTP (`wget` / `curl`)**

The [dataset card](https://huggingface.co/datasets/neashton/drivaerml) documents looping over `run_$i` and pulling individual files. For **surface** workflow smoke tests, fetch at least **`boundary_<i>.vtp`** per run. Example (adjust the ID list or `seq` range):

```bash
HF_OWNER="neashton"
HF_PREFIX="drivaerml"
LOCAL_DIR="./drivaer_data"
mkdir -p "$LOCAL_DIR"

for i in 1 11 202; do
  RUN_DIR="run_${i}"
  DEST="$LOCAL_DIR/$RUN_DIR"
  mkdir -p "$DEST"
  wget -c "https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main/${RUN_DIR}/boundary_${i}.vtp" \
    -O "$DEST/boundary_${i}.vtp"
done
```

Then point the benchmark at the parent directory:

```bash
python main.py benchmark.datasets.0.root="$(pwd)/drivaer_data"
```

Adapt the URL pattern to add `volume_*` parts, STL, or force CSVs as needed; the card’s **Example 2** shows the same idea for STL + force files.

**3) Hugging Face Hub CLI — partial tree**

Install the CLI and prefer **include patterns** so you do not sync 31 TB by accident:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download neashton/drivaerml --repo-type dataset \
  --local-dir /path/to/drivaerml_subset \
  --include "run_1/*" --include "run_11/*"
```

Expand `--include` for every `run_<i>` you need. For a full local mirror without Git, understand the size implications before omitting filters.

**NGC:** when the [benchmark table](#benchmark-assets-on-ngc-coming-soon) lists a DrivAerML bundle, unpack so the **`run_*`** layout matches the table above and set **`benchmark.datasets[].root`** accordingly.

**Train / validation split for benchmarking**

After the tree is on disk, use the fixed 90/10 split files under [`drivaer_ml_files/`](drivaer_ml_files/) if you want to align case lists with that split — see [`drivaer_ml_files/README.md`](drivaer_ml_files/README.md).

---

## Notebooks (exploration, not batch production)

The **`notebooks/`** directory (`surface_benchmarking.ipynb`, `volume_benchmarking.ipynb`, **`adding_a_new_model.ipynb`**, etc.) supports **interactive** exploration and onboarding; **`main.py`** + **`conf/`** remains the recommendation for **logged, repeatable** benchmark runs aligned with CI and matrix configs.

---

## Exit codes (automation)

- By default, **`main.py`** exits **0** when `run_benchmark` completes without raising.
- Set **`run.fail_on_all_skipped: true`** to exit **1** if every model×dataset run was skipped (e.g. domain mismatch in matrix mode).
- Set **`run.fail_on_any_metric_nan: true`** to exit **1** if any aggregate metric in `metrics` is NaN.
- Override on the CLI: `run.fail_on_all_skipped=true`.

The flat CLI **`python -m physicsnemo.cfd.evaluation.benchmarks.run`** applies the same policy.

---

## Config files (short reference)

| File | Role |
| ---- | ----- |
| [`conf/config_matrix_surface_custom.yaml`](conf/config_matrix_surface_custom.yaml) | **Matrix surface**, **local** `checkpoint` / `stats_path`; **`run.output_dir`**: **`benchmark_results_matrix_surface`**. |
| [`conf/config_matrix_volume_custom.yaml`](conf/config_matrix_volume_custom.yaml) | **Matrix volume**, same for VTU runs; **`run.output_dir`**: **`benchmark_results_matrix_volume`**. |
| [`conf/config_matrix_surface_hf.yaml`](conf/config_matrix_surface_hf.yaml) | **Matrix surface**, **HF** assets; **`run.output_dir`**: **`benchmark_results_matrix_surface_hf`**. Default **`python main.py`**. |
| [`conf/config_matrix_volume_hf.yaml`](conf/config_matrix_volume_hf.yaml) | **Matrix volume**, **HF** assets; **`run.output_dir`** as in file. |

**Matrix mode:** every **`benchmark.models`** entry runs against every **`benchmark.datasets`** entry; incompatible surface/volume pairs are skipped. Edit checkpoint paths in **`_custom`**; comment out any **`- name:`** block you do not need.

---

## Config reference (YAML keys)

| Section | Contents |
| ------- | -------- |
| **run** | `device`, `output_dir`, `seed`, `batch_size`, **`save_inference_mesh`**, **`distributed`**, **`fail_on_all_skipped`**, **`fail_on_any_metric_nan`**, optional **`metrics_cache`** |
| **model** / **benchmark.models** | `name`, `checkpoint`, `stats_path`, optional **`package`** (`hf://org/repo@rev`, `s3://…`, or local dir), **`checkpoint_relpath`** / **`stats_relpath`** (paths inside the package), `inference_domain`, `kwargs` |
| **dataset** / **benchmark.datasets** | `name`, `root`, optional `case_ids` (`null` = all cases from `adapter.list_cases()`), `kwargs` |
| **output** | VTK field name maps; **`streamlines_vector_canonical`** (volume); optional **`surface_interpolate_point_to_cell_for_metrics`** (kNN-IDW point → cell for XmGN/FiGNet-style surfaces so drag/lift/L2 use **`metric_dtype: cell`**), plus **`surface_metrics_idw_k`**, **`surface_metrics_idw_device`** |
| **metrics** | Metric names or `{ name: ..., ...kwargs }` |
| **reports** | Optional PNG pipeline |
| **benchmark** | `mode`, `models` / `datasets`, **`reproducibility`** (`log_env`, `save_artifacts`) |

**`case_id`:** optional top-level Hydra key: `null` uses each dataset's `case_ids` (or all adapter cases); a **string** runs one case on every dataset; a **list** runs those cases in matrix mode.

**`benchmark.reproducibility.log_env`:** when `true`, writes **full `os.environ`** to `env.json` under `run.output_dir` — avoid in shared CI or when HF/AWS/other secrets may be present (default **`false`** in code and **in all example YAMLs under `conf/`**). Enable only deliberately for short-lived local debugging.

**Metrics cache:** `run.metrics_cache.enabled` stores per-case scalars; delete the cache directory for a full recompute. Plots and meshes are not cached.

**Remote model assets:** Install optional **`pip install 'nvidia-physicsnemo-cfd[evaluation-hf]'`** for `hf://` and `s3://` package roots. Cache directory defaults to `~/.cache/physicsnemo-cfd/models` or override with **`PHYSICSNEMO_CFD_MODEL_CACHE`**. Built-in matrix models (including **domino**, which also resolves **`domino_config`** from the package) use per-model roots in **`physicsnemo.cfd.evaluation.assets.builtin_packages`** when **`checkpoint`** / **`stats_path`** are omitted; override with explicit paths or **`model.package`** as needed. **`register_default_asset`** remains available for custom names. See **[CONTRIBUTING.md](../../CONTRIBUTING.md)** for custom-wrapper tiers.

---

## Custom models, datasets, and metrics

### Custom models

1. Subclass **`CFDModel`** (`physicsnemo.cfd.evaluation.models.model_registry`) under `physicsnemo/cfd/evaluation/models/wrappers/`.
2. Implement `INFERENCE_DOMAIN`, `OUTPUT_LOCATION`, `load`, `prepare_inputs`, `predict`, `decode_outputs(raw_output, case, model_input=None)` (the engine passes the same `model_input` returned by `prepare_inputs`).
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

Registered names (or dicts with `name` + kwargs). Surface and volume matrix configs use different metric subsets — **`drag`** and **`lift`** are registered for **`domain="surface"`** only (`evaluation/metrics/builtin/forces.py`). Do **not** list **`drag`** / **`lift`** under volume benchmarks: **`get_metric`** has no volume entry, and the benchmark would record NaNs without a loud error.

**Surface example:**

```yaml
metrics:
  - l2_pressure
  - l2_shear_stress
  - drag
  - lift
```

**Volume example** (`config_matrix_volume_custom.yaml` / `config_matrix_volume_hf.yaml`):

```yaml
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
| `l2_pressure_area_weighted` | Area-weighted L2 pressure |
| `drag`, `lift` | Coefficient errors (surface inference only); expands to `drag_error`, etc. |
| `l2_pressure`, `l2_velocity`, `l2_turbulent_viscosity` (volume) | Volume-field L2 (see **`l2.py`**, `domain="volume"`) |
| `continuity_residual_l2`, `momentum_residual_l2` | Volume residuals (volume configs) |

---

## Reports and plots

When **`reports.enabled`** and **`reports.visuals`** are set, PNGs are written under `{run.output_dir}/visuals/`. Comparison VTK exists if **`reports.save_comparison_meshes: true`**.

When **`reports.visuals`** is set, each registered visual receives the full **`results`** list (one entry per model × dataset cell); built-ins iterate runs and write `{run.output_dir}/visuals/` PNGs with **`model`** / **`dataset`** in the filename where applicable.

**`aggregate_volume_errors`** writes per-run files such as **`{model}_{dataset}_aggregate_resampled_volume.vtk`** and **`{model}_{dataset}_aggregate_volume_*_slice.png`** (sanitized names). Headless servers may need **xvfb** (see **`setup.sh`** for an example `apt` line — optional).

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

Smoke-test wrappers: **`checkpoint: ""`**, **`stats_path: ""`**. Use under **`benchmark.models`** entries.

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
