# DoMINO Sensitivity Analysis for Aerodynamic Design

This workflow computes geometry sensitivities for external aerodynamics
using a pre-trained DoMINO surrogate. Given a surface mesh and flow
conditions, it predicts the surface pressure and wall shear stress
fields and returns gradients of total drag with respect to the mesh
coordinates. The resulting sensitivity vectors indicate the direction
each surface element should move to reduce drag.

Highlights:

- Surface-normal sensitivity maps that show where adding or removing
  material reduces drag.
- Post-processing utilities for Laplacian smoothing and normal
  projection of the raw gradients.
- Finite-difference validation of the autograd gradient.
- Batched, optionally multi-GPU inference for large meshes.

## Contents

- `main.py`: `DoMINOInference` pipeline and post-processing utilities.
- `main.ipynb`: End-to-end notebook walkthrough.
- `gradient_checking.ipynb`: Finite-difference validation of the
  computed sensitivities.
- `design_datapipe.py`: Mesh preprocessing and neighborhood
  construction for the model's input dict.
- `utilities/mesh_postprocessing.py`: Laplacian smoothing on the
  surface mesh.
- `conf/config.yaml`: Minimal Hydra config (just the spatial bounding
  boxes used to crop and normalize the input geometry).
- `geometries/`: Sample STLs.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU
- Workflow-specific extras:

```bash
pip install -r requirements.txt --no-build-isolation
```

## Get the checkpoint bundle

The pre-trained checkpoint is published as a bundle on Hugging Face:

[nvidia/domino_drivaerml/domino_drivaerml_surface_checkpoint](https://huggingface.co/nvidia/domino_drivaerml/tree/main/domino_drivaerml_surface_checkpoint)

Download these files into a new `checkpoints/` directory next to
`main.py`:

| File | Purpose |
| --- | --- |
| `DoMINO.0.501.mdlus` | Model architecture **and** trained weights. Self-describing - the file knows its own hyperparameters. |
| `scaling_factors.pkl` | Per-field min/max statistics from training; used to un-normalize the model outputs back to physical units. |
| `config.yaml` | Training-time config. Only the spatial bounding boxes (`data.bounding_box{,_surface}`) are needed at inference; copy them into `conf/config.yaml` if you change datasets. |
| `global_stats.json` | Summary statistics about the training data. Not read at inference; included for reference. |

The HF directory also contains a `checkpoint.0.501.pt` (a trainer-resume
artifact with optimizer state). It is **not** used by this workflow;
you can skip it unless you plan to resume training.

After download, your tree should look like:

```
workflows/domino_design_sensitivities/
├── main.py
├── conf/config.yaml
└── checkpoints/
    ├── DoMINO.0.501.mdlus
    ├── scaling_factors.pkl
    ├── config.yaml          # (optional, for reference)
    └── global_stats.json    # (optional, for reference)
```

## Run

From inside `workflows/domino_design_sensitivities/`:

```bash
uv run main.py
```

By default this loads `./checkpoints/DoMINO.0.501.mdlus`, downloads
the sample DrivAerML STL on first run, runs the model, computes the
drag sensitivity, applies Laplacian smoothing, and writes the result
as a `.vtk` next to the input STL. Both `--model-checkpoint-path` and
`--input-file` can be overridden on the command line.

## API

```python
from main import DoMINOInference

domino = DoMINOInference(
    cfg=cfg,                                                # see `conf/config.yaml`
    model_checkpoint_path="./checkpoints/DoMINO.0.501.mdlus",
    dist=DistributedManager(),                              # optional; single-GPU if omitted
)

results = domino(
    mesh=mesh,                  # pv.PolyData surface mesh
    stream_velocity=38.889,     # m/s
    stencil_size=7,             # surface neighborhood size
    air_density=1.205,          # kg/m^3
    verbose=True,               # show batch progress
)
```

`DoMINOInference.__post_init__` calls `DoMINO.from_checkpoint(...)`,
so the loaded model architecture matches the checkpoint exactly - the
local `cfg` does not need to know any model hyperparameters. The
unnormalization factors are read lazily from `scaling_factors.pkl`
next to the `.mdlus`.

`results` is a `dict[str, np.ndarray]`; see per-field shapes below:

- `geometry_coordinates`: (n_points, 3) sampled mesh point coordinates
- `geometry_sensitivity`: (n_points, 3) raw d(-drag)/dX vectors
- `pred_surf_pressure`: (n_faces,) surface pressure [Pa]
- `pred_surf_wall_shear_stress`: (n_faces, 3) wall shear stress [Pa]
- `aerodynamic_force`: (3,) total integrated force [Fx, Fy, Fz] [N]

The gradient is of -drag, so vectors point in the direction that
**reduces** drag when the surface moves along them. Batching is
handled internally; if you see OOM, reduce `stencil_size` or decimate
the mesh.

### Post-processing

```python
processed = DoMINOInference.postprocess_point_sensitivities(
    results=results,
    mesh=mesh,                  # pv.PolyData with normals
    n_laplacian_iters=20,
)
```

Adds the following fields:

- `raw_sensitivity_cells`: (n_faces, 3) raw vectors d(-drag)/dX
- `raw_sensitivity_normal_cells`: (n_faces,) projection onto cell normals
- `smooth_sensitivity_point`: (n_points, 3) Laplacian-smoothed vector field
- `smooth_sensitivity_normal_point`: (n_points,) Laplacian-smoothed normal component
- `smooth_sensitivity_cell`: (n_faces, 3) point-smoothed field transferred to cells
- `smooth_sensitivity_normal_cell`: (n_faces,) point-smoothed normal component on cells

Smoothing uses `utilities/mesh_postprocessing.py` (CSR adjacency and
Numba-accelerated Laplacian averaging on the 1-ring). Larger
`n_laplacian_iters` means stronger smoothing.

## Configuration

`conf/config.yaml` contains only what the inference pipeline cannot
infer from the checkpoint:

- `data.bounding_box` and `data.bounding_box_surface`: min/max corners
  for volume and surface sampling. **These must match the training
  domain of your checkpoint.** Use the values from the HF
  `config.yaml` if you swap in a different checkpoint.
- `variables.surface.solution`: pure documentation of the channel
  layout returned by the model (column 0 is pressure, columns 1-3 are
  wall shear stress). Not consumed by `DoMINOInference`.

Everything that used to live here - model architecture, neighbor
counts, basis-function sizes, normalization factors - now lives in the
checkpoint bundle. Don't re-add it to `conf/config.yaml`: any drift
between the local config and the checkpoint's actual training config
would not silently break the run, but it would mislead future readers.

The notebooks initialize Hydra like this:

```python
with hydra.initialize(version_base="1.3", config_path="./conf"):
    cfg = hydra.compose(config_name="config")
```

## Bring your own checkpoint

If you train your own DoMINO with the upstream
[physicsnemo DoMINO example](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/domino):

1. Save the trained model via
   `physicsnemo.Module.save("Mine.mdlus")`. This writes a
   self-describing archive (architecture args + weights), so it works
   with `DoMINO.from_checkpoint("Mine.mdlus")` here without any code
   changes.
2. Copy the `scaling_factors.pkl` produced by
   `compute_statistics.py` (in the training example) next to your
   `.mdlus`. `DoMINOInference` looks for it at
   `<.mdlus>.parent / "scaling_factors.pkl"`.
3. Update `conf/config.yaml`'s `data.bounding_box{,_surface}` to
   match the bounds you used during training.

Then point `--model-checkpoint-path` at your new file.

## Gradient checking (finite differences)

See `gradient_checking.ipynb`. Outline:

1. Run a baseline inference on the sample mesh.
2. Post-process to get the raw and smoothed sensitivity fields.
3. Perturb point coordinates by `epsilon * sensitivities` and
   re-evaluate drag.
4. Sweep symmetric `epsilon` values across orders of magnitude.
5. Compare the finite-difference drag deltas to the autograd
   prediction on symlog axes.

Tips:

- This sweep is heavy (many forward evaluations on the full mesh). To
  iterate faster, trim the `epsilons` list to a few orders of magnitude
  or reduce `stencil_size`.
- Smoothed normal sensitivities produce more stable finite-difference
  behavior than the raw vector field.

## Visualization

```python
mesh.plot(
    scalars="smooth_sensitivity_normal_cell",
    cmap="RdBu_r",
    jupyter_backend="static",
    cpos=[-1, -1, 1],
    clim=(-1, 1),
)
```

Warping for intuition (illustration only - not a physical deformation):

```python
warped = mesh.warp_by_scalar("smooth_sensitivity_normal_point", factor=0.05)
warped.plot(scalars="smooth_sensitivity_normal_cell", cmap="RdBu_r")
```

## Notes and guidance

- Sensitivities are valid for small, smooth deformations. Large warps
  are for visualization only.
- Projection to surface normals removes tangential components that
  should not affect the underlying PDE solution.
- For multi-GPU, `DistributedManager` is initialized automatically; a
  single-process run falls back to single-GPU/CPU.

### Limitations and model smoothness (C1 continuity)

Design sensitivities assume the surrogate is at least C1 (once
continuously differentiable) with respect to inputs that affect
geometry and flow conditions. If the model is not sufficiently
smooth, gradients can be noisy, unstable, or misleading.

Actionable configuration guidance:

- **Activation functions**: Prefer smooth choices with continuous
  derivatives, such as SiLU/Swish, GELU, or Softplus (with beta > 1
  for sharper yet smooth transitions). Avoid ReLU/LeakyReLU/PReLU if
  you require derivative continuity at zero; those are only piecewise
  linear and not C1.
- **Neighborhood/aggregation**: Hard top-k selections and max pooling
  are non-differentiable at selection boundaries. Keep the stencil
  fixed during sensitivity evaluation and restrict deformations to be
  small. When possible, prefer smooth, density-weighted aggregations
  (e.g., softmax-weighted sums, averages) over hard maxima.
- **Geometry encodings**: Use continuous encodings (e.g., signed
  distance fields and smooth positional transforms). Avoid non-smooth
  ops like `abs`, `floor`, or conditional kinks in geometry
  pipelines. Fourier features are smooth.
- **Data preprocessing**: Neighborhood graph construction (e.g.,
  k-NN) is a discrete operation. Gradients are valid locally under
  fixed connectivity, but can jump when connectivity changes. Keep
  `stencil_size` fixed and geometry perturbations small for gradient
  checks and early-stage optimization.
- **Post-processing**: The Laplacian smoothing and normal projection
  are for interpretation. They are not in the gradient path used to
  compute `geometry_sensitivity`. Use them to regularize updates if
  you couple sensitivities to a design loop.

In short: choose smooth activations, avoid hard discontinuities in
pre-processing/post-processing, and keep perturbations small so that
the piecewise-smooth assumptions remain valid.

## References

- DoMINO: A Decomposable Multi-scale Iterative Neural Operator:
  [arXiv:2501.13350](https://arxiv.org/abs/2501.13350)
- Automatic Differentiation in Machine Learning: A Survey:
  [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)
