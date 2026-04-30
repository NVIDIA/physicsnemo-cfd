---
name: create-model-wrapper
description: >-
  Create a new model wrapper for the PhysicsNeMo CFD benchmarking workflow.
  Use when the user wants to add a new CFD model, write a CFDModel wrapper,
  integrate a new neural network architecture, or run a custom model through
  the benchmarking pipeline.
---

# Create a Model Wrapper

Guide the user through adding a new CFD model to the benchmarking workflow by writing a `CFDModel` subclass.

## Reference files to read first

Before starting, read these files for context:

- `physicsnemo/cfd/evaluation/models/model_registry.py` — base class and registry
- `physicsnemo/cfd/evaluation/datasets/schema.py` — `CanonicalCase`, `build_predictions_dict`
- `physicsnemo/cfd/evaluation/models/wrappers/surface_baseline.py` — simplest concrete wrapper
- `physicsnemo/cfd/evaluation/models/wrappers/__init__.py` — how wrappers are registered
- `physicsnemo/cfd/evaluation/common/io.py` — mesh loading and normalization stats helpers
- `workflows/benchmarking_workflow/notebooks/adding_a_new_model.ipynb` — end-to-end tutorial

## The `CFDModel` interface

Every wrapper must set two class variables and implement four methods:

| Member | Purpose |
|--------|---------|
| `INFERENCE_DOMAIN` | `"surface"` or `"volume"` — which mesh manifold |
| `OUTPUT_LOCATION` | `"point"` or `"cell"` — where predictions live on the mesh |
| `output_location` (property) | Instance-level access to `OUTPUT_LOCATION` |
| `load(checkpoint_path, stats_path, device, **kwargs)` | Load weights and stats; return `self` |
| `prepare_inputs(case: CanonicalCase)` | Convert canonical case into model-specific tensors/graphs |
| `predict(model_input)` | Run forward pass; return raw output |
| `decode_outputs(raw_output, case, model_input=None)` | Denormalize and map to canonical predictions dict |

The engine calls `load` once, then `prepare_inputs → predict → decode_outputs(raw, case, model_input)` per case (`model_input` is the dict from `prepare_inputs`; use when decode must mirror inference geometry).

## Step 1: Write the wrapper class

```python
import numpy as np
import torch
import pyvista as pv
from typing import Any, ClassVar

from physicsnemo.cfd.evaluation.models.model_registry import (
    CFDModel, register_model, OutputLocation,
)
from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase, InferenceDomain, build_predictions_dict,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference


class MyModelWrapper(CFDModel):
    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"  # or "volume"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"       # or "point"

    def __init__(self) -> None:
        self._model = None
        self._stats = None
        self._device = "cpu"

    @property
    def output_location(self) -> OutputLocation:
        return self.OUTPUT_LOCATION

    def load(self, checkpoint_path, stats_path, device, **kwargs):
        self._device = device
        # Load your model architecture + weights
        # self._model = ...
        # Load normalization stats if needed
        # self._stats = ...
        log_inference("my_model", f"Loaded from {checkpoint_path}")
        return self

    def prepare_inputs(self, case: CanonicalCase):
        mesh = pv.read(case.mesh_path)
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()
        # Extract coordinates and build model-specific input
        # (tensors, graphs, point clouds, etc.)
        coords = np.array(mesh.cell_centers().points, dtype=np.float32)
        return torch.tensor(coords, device=self._device)

    def predict(self, model_input):
        # Run forward pass through your model
        with torch.no_grad():
            raw_output = self._model(model_input)
        return raw_output

    def decode_outputs(self, raw_output, case, model_input=None):
        # Denormalize if needed. Surface: pressure + shear_stress; volume: add velocity, turbulent_viscosity.
        return build_predictions_dict(
            pressure=raw_output["pressure"].cpu().numpy(),
            shear_stress=raw_output["shear_stress"].cpu().numpy(),
        )
```

### Key implementation considerations

**Normalization**: Most trained models normalize inputs/outputs. Load stats from `stats_path` in `load()` and denormalize in `decode_outputs()`. See `physicsnemo/cfd/evaluation/common/io.py` for `load_global_stats()` and related helpers.

**Batching**: For large meshes, `prepare_inputs` may need to subsample or batch. Use `kwargs` passed through `load()` (e.g., `batch_resolution`, `geometry_sampling`) to control this.

**Output shape**: `pressure` must be `(N,)` float32. `shear_stress` must be `(N, 3)` float32 for surface. Volume fields: `velocity` is `(N, 3)`, `turbulent_viscosity` is `(N,)`.

**Output location**: If `OUTPUT_LOCATION = "cell"`, return N = `mesh.n_cells` values. If `"point"`, return N = `mesh.n_points` values.

## Step 2: Create checkpoint and stats files

Your model needs a checkpoint file and optionally a `global_stats.json`:

```python
# Checkpoint: save your model's state dict
torch.save(model.state_dict(), "checkpoint.pt")

# Stats: JSON with mean/std_dev for denormalization
# Surface format:
{
    "mean": {"pressure": [0.0], "shear_stress": [0.0, 0.0, 0.0]},
    "std_dev": {"pressure": [1.0], "shear_stress": [1.0, 1.0, 1.0]}
}
# Volume format:
{
    "mean": {"pressure": [0.0], "velocity": [0.0, 0.0, 0.0], "turbulent_viscosity": [0.0]},
    "std_dev": {"pressure": [1.0], "velocity": [1.0, 1.0, 1.0], "turbulent_viscosity": [1.0]}
}
```

## Step 3: Register and test

```python
register_model("my_model", MyModelWrapper)

# Load a case from any registered dataset adapter
from physicsnemo.cfd.evaluation.datasets.adapters.drivaerml import DrivAerMLAdapter
adapter = DrivAerMLAdapter(root="/path/to/data", inference_domain="surface")
case = adapter.load_case(adapter.list_cases()[0])

# Run the full inference pipeline
wrapper = MyModelWrapper()
wrapper.load(checkpoint_path="checkpoint.pt", stats_path="global_stats.json", device="cuda:0")
model_input = wrapper.prepare_inputs(case)
raw_output = wrapper.predict(model_input)
predictions = wrapper.decode_outputs(raw_output, case, model_input)

assert "pressure" in predictions
assert predictions["pressure"].shape[0] > 0
```

## Step 4: Run the full benchmark

```python
from physicsnemo.cfd.evaluation.config import Config
from physicsnemo.cfd.evaluation.benchmarks.engine import run_benchmark

config = Config.from_dict({
    "run": {"device": "cuda:0", "output_dir": "results", "metrics_cache": {"enabled": False}},
    "benchmark": {
        "mode": "matrix",
        "models": [{
            "name": "my_model",
            "inference_domain": "surface",
            "checkpoint": "/path/to/checkpoint.pt",
            "stats_path": "/path/to/global_stats.json",
            "kwargs": {},
        }],
        "datasets": [{
            "name": "drivaerml",
            "root": "/path/to/drivaerml/data",
            "case_ids": ["run_1", "run_11"],
            "kwargs": {"align_ground_truth_to_model": True, "inference_domain": "surface"},
        }],
        "reproducibility": {"log_env": False, "save_artifacts": True},
    },
    "output": {"mesh_field_names": {"pressure": "pMeanTrimPred", "shear_stress": "wallShearStressMeanTrimPred"}},
    "metrics": ["l2_pressure", "l2_shear_stress", "l2_pressure_area_weighted", "drag", "lift"],
    "reports": {"enabled": False},
})
results = run_benchmark(config)
```

Results are written to `benchmark_results.json` (a JSON list of dicts, one per model×dataset combo).

## Step 5: Visualize predictions

```python
from physicsnemo.cfd.postprocessing_tools.visualization.utils import plot_fields, plot_field_comparisons

# Just the predicted fields (no GT comparison):
plotter = plot_fields(mesh, fields=["pMeanTrimPred"], view="xy", dtype="cell", window_size=[1800, 600])
plotter.screenshot("predicted_pressure.png")
plotter.close()

# Side-by-side with GT (GT | Pred | Error):
plotter = plot_field_comparisons(mesh, true_fields=["pMeanTrim"], pred_fields=["pMeanTrimPred"],
                                  view="xy", dtype="cell", window_size=[1800, 600])
plotter.screenshot("comparison.png")
plotter.close()
```

## Step 6: Make permanent (optional)

Save the wrapper to `physicsnemo/cfd/evaluation/models/wrappers/my_model.py` and register in `wrappers/__init__.py`:

```python
from physicsnemo.cfd.evaluation.models.wrappers.my_model import MyModelWrapper
register_model("my_model", MyModelWrapper)
```

Then use `model.name: my_model` in any YAML config.

## Gotchas

- **DistributedManager**: Model wrappers may call `DistributedManager.initialize()`. In notebooks without `torchrun`, set env vars first: `WORLD_SIZE=1`, `RANK=0`, `LOCAL_RANK=0`, `MASTER_ADDR=localhost`, `MASTER_PORT=12355`.
- **`weights_only=True`**: Use this flag with `torch.load()` for safe deserialization (PyTorch 2.6+ default).
- **Domain matching**: The engine checks that `model.INFERENCE_DOMAIN` matches the dataset adapter's `inference_domain_from_kwargs()`. Mismatches are skipped in matrix mode or raise in single mode.
- **GT alignment**: When `align_ground_truth_to_model: true` in dataset kwargs, the engine converts GT data to match `OUTPUT_LOCATION` (point ↔ cell). This is automatic — the wrapper just needs correct class vars.
- **Results JSON format**: `benchmark_results.json` is a plain `list[dict]`, not `{"results": [...]}`. Iterate directly: `for combo in report:`.
