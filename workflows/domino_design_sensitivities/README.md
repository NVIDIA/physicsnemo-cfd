## DoMINO Sensitivity Analysis for Aerodynamic Design

This workflow demonstrates how to compute and visualize geometry sensitivities for external aerodynamics using a pre-trained DoMINO surrogate. It provides:

- **Sensitivity visualization**: surface-normal sensitivity maps highlighting where adding/removing material reduces drag
- **Postprocessing**: Laplacian smoothing and normal projection for physically meaningful fields
- **Validation**: finite-difference gradient checking against adjoint/autograd gradients
- **Batching and multi-GPU**: efficient inference for large meshes

### What this computes
Given a surface mesh and flow conditions, the pipeline predicts surface pressure and wall shear stress and computes gradients of total drag with respect to the mesh coordinates. The returned sensitivity vectors indicate the direction to move each surface element to reduce drag.


## Contents

- `main.py`: Core inference pipeline with `DoMINOInference` and postprocessing utilities
- `main.ipynb`: End-to-end walkthrough for running inference, postprocessing, and visualization
- `gradient_checking.ipynb`: Finite-difference validation of sensitivities via geometry perturbations
- `design_datapipe.py`: Mesh preprocessing and neighborhood construction for inference
- `utilities/mesh_postprocessing.py`: Laplacian smoothing implementation
- `conf/config.yaml`: Hydra configuration (domain bounds, variables, model params)
- `geometries/`: Sample meshes (`drivaer_1_single_solid.stl`, decimated variant, and generated `.vtk`)


## Prerequisites

- Python 3.10+
- CUDA-capable GPU
- Packages: in addition to the base packages required by PhysicsNeMo-CFD, you'll need those in `requirements.txt`. These can be installed with:

```bash
pip install -r requirements.txt --no-build-isolation
```

You also need a pre-trained DoMINO checkpoint, which, for space reasons, is not included in this repository. Replace `DoMINO.0.41.pt` with your own model (see the [DoMINO training example](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/domino)).


## Quick start

Open `main.ipynb` and run all cells. It covers:
- Config and distributed setup (Hydra + `DistributedManager`)
- Loading a geometry with PyVista
- Running `DoMINOInference` to get results and sensitivities
- Postprocessing to surface-normal and smoothed fields
- Visualization and warping for intuition

## API

### DoMINOInference
```python
results = DoMINOInference(
    cfg=cfg,  # DictConfig from Hydra; see conf/config.yaml
    model_checkpoint_path="./DoMINO.0.41.pt",  # or your checkpoint path
    dist=DistributedManager(),  # optional; single-GPU if omitted
)(
    mesh=mesh,                 # pv.PolyData surface mesh
    stream_velocity=38.889,    # m/s
    stencil_size=7,            # surface neighborhood size
    air_density=1.205,         # kg/m^3
    verbose=True,              # show batch progress
)
```

Returns a `dict[str, np.ndarray]` with shapes referenced to surface cells unless noted:
- `geometry_coordinates`: (n_points, 3) sampled coordinates used internally
- `geometry_sensitivity`: (n_faces, 3) raw sensitivity vectors d(-drag)/dX
- `pred_surf_pressure`: (n_faces,) surface pressure [Pa]
- `pred_surf_wall_shear_stress`: (n_faces, 3) wall shear [Pa]
- `aerodynamic_force`: (3,) [Fx, Fy, Fz] [N]

Notes:
- The gradient is taken of -drag, so the vectors point in the direction that reduces drag when moving the surface.
- Batching is handled internally. If you see OOM, reduce neighborhood size or decimate the mesh.

### Postprocessing
```python
processed = DoMINOInference.postprocess_point_sensitivities(
    results=results,
    mesh=mesh,                  # pv.PolyData with normals
    n_laplacian_iters=20,
)
```
Adds commonly used sensitivity fields (keys and shapes):
- `raw_sensitivity_cells`: (n_faces, 3) raw sensitivity vectors d(-drag)/dX
- `raw_sensitivity_normal_cells`: (n_faces,) projection onto cell normals
- `smooth_sensitivity_point`: (n_points, 3) smoothed vector field on points
- `smooth_sensitivity_normal_point`: (n_points,) smoothed scalar normal component on points
- `smooth_sensitivity_cell`: (n_faces, 3) point-smoothed field transferred to cells
- `smooth_sensitivity_normal_cell`: (n_faces,) point-smoothed scalar transferred to cells

Smoothing uses `utilities/mesh_postprocessing.py` (CSR adjacency + Numba-accelerated Laplacian averaging on the 1-ring). Increase `n_laplacian_iters` for stronger smoothing.


## Configuration

`conf/config.yaml` defines the domain, variables, and model hyperparameters. Important entries:
- `data.bounding_box` and `data.bounding_box_surface`: min/max corners for volume and surface sampling. Should be consistent with training data bounds.
- `variables.surface.solution` and `variables.volume.solution`: names and types (scalar/vector) that determine output channels.
- `model.interp_res`, `model.num_surface_neighbors`, and related geometry extraction parameters control memory/performance.

The notebooks initialize Hydra like this:
```python
with hydra.initialize(version_base="1.3", config_path="./conf"):
    cfg = hydra.compose(config_name="config")
```


## Gradient checking (finite differences)
See `gradient_checking.ipynb` for a full walk-through. Outline:

1. Run baseline inference on a decimated mesh for speed: `geometries/drivaer_1_single_solid_decimated3.stl`.
2. Postprocess to obtain raw and smoothed sensitivity fields.
3. Define `get_drag(epsilon, sensitivities)` that perturbs point coordinates by `epsilon * sensitivities` and re-evaluates drag.
4. Sweep symmetric `epsilon` values over several orders of magnitude and compute drag deltas for both raw and smoothed fields.
5. Plot drag change vs. epsilon on symlog axes and compare against the adjoint (autograd) linear prediction.

Tips:
- This is computationally heavy (many forward evaluations). Use the decimated mesh and consider limiting the epsilon set for quick checks.
- Smoothed normal sensitivities often produce more stable finite-difference behavior.


## Visualization
The notebooks use PyVista to visualize scalar fields like `smooth_sensitivity_normal_cell`:

```python
mesh.plot(
    scalars="smooth_sensitivity_normal_cell",
    cmap="RdBu_r",
    jupyter_backend="static",
    cpos=[-1, -1, 1],
    clim=(-1, 1),
)
```

Warping for intuition (purely illustrative):
```python
warped = mesh.warp_by_scalar("smooth_sensitivity_normal_point", factor=0.05)
warped.plot(scalars="smooth_sensitivity_normal_cell", cmap="RdBu_r")
```


## Notes and guidance
- Sensitivities are valid for small, smooth deformations. Large warps are for visualization only.
- Projection to surface normals removes tangential components that should not affect the PDE solution.
- For multi-GPU, `DistributedManager` is initialized automatically; if a single process is detected, it runs in single-GPU/CPU mode.
- If you trained your own model, update the checkpoint path and ensure config bounds match your training domain.


## References
- DoMINO: A Decomposable Multi-scale Iterative Neural Operator: [arXiv:2501.13350](https://arxiv.org/abs/2501.13350)
- Automatic Differentiation in Machine Learning: A Survey: [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)
