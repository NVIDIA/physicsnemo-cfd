# NIM inference (DoMINO) + DrivAerML benchmarking

This workflow holds **tutorial notebooks** that call the **DoMINO Automotive
Aerodynamics**
[NVIDIA Inference Microservice (NIM)](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/domino-automotive-aero)
from Python, map predictions onto **DrivAerML** VTK meshes, and compute metrics
with **`physicsnemo.cfd.postprocessing_tools`**.

It complements the
**[Hydra benchmarking workflow](../benchmarking/README.md)** at
`workflows/benchmarking/`, which runs packaged checkpoints (for example from
Hugging Face) via `python main.py`. Use **this folder** when you already have
the NIM API and want a **notebook walkthrough**; use **benchmarking** for
matrix evaluation and CI-style configs.

## Contents

| Path | Description |
|------|-------------|
| [`notebooks/surface_benchmarking.ipynb`](notebooks/surface_benchmarking.ipynb) | Surface VTP: NIM → interpolate to cell centers → L2 / area-weighted L2, drag–lift, plots, streamlines. |
| [`notebooks/volume_benchmarking.ipynb`](notebooks/volume_benchmarking.ipynb) | Volume VTU: NIM → interpolate to points → L2, error vs. SDF, slices, optional physics checks. |
| [`notebooks/benchmarking_in_absence_of_gt.ipynb`](notebooks/benchmarking_in_absence_of_gt.ipynb) | No ground-truth setting: STL resolution variants, NIM ensemble / variance, Chamfer–Hausdorff geometry distances. |

## Prerequisites

1. **PhysicsNeMo-CFD** installed from the repository root (see the
   [main README](../../README.md#installation)).
2. **DoMINO NIM** running and reachable (default in the notebooks:
   `http://localhost:8000/v1/infer`). Follow the
   [NIM quickstart](https://docs.nvidia.com/nim/physicsnemo/domino-automotive-aero/latest/quickstart-guide.html).
3. **GPU** recommended for interpolation and large VTU loads; volume notebooks
   assume multi‑GB meshes.

If the notebook runs in **Docker** alongside the NIM, start both with
**`--network host`** so localhost inference works.

## API used in the notebooks

- **`physicsnemo.cfd.evaluation.nims.call_domino_nim`** — HTTP client for the
  NIM (see [`domino_nim.py`](../../physicsnemo/cfd/evaluation/nims/domino_nim.py)).
- **`physicsnemo.cfd.postprocessing_tools`** — interpolation onto meshes, L2
  and area-weighted metrics, forces, visuals, streamlines, etc.

## Data

Examples download sample **run_202** assets from the
[DrivAerML Hugging Face dataset](https://huggingface.co/datasets/neashton/drivaerml).
Volume VTUs may be split into `.part` files on the Hub; the notebook
concatenates them before reading.

## Legacy location

These notebooks were previously under
**`workflows/deprecated/bench_example/notebooks/`**. They are maintained here
so **NIM-based** tutorials live next to **`workflows/benchmarking/`** without
mixing into the deprecated layout.
