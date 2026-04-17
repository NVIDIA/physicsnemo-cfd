---
name: create-dataset-adapter
description: >-
  Create a new dataset adapter for the PhysicsNeMo CFD benchmarking workflow.
  Use when the user wants to add a new CFD dataset, write a DatasetAdapter,
  integrate a new mesh format, or benchmark models on custom data.
---

# Create a Dataset Adapter

Guide the user through adding a new CFD dataset to the benchmarking workflow by writing a `DatasetAdapter` subclass.

## Reference files to read first

Before starting, read these files for context:

- `physicsnemo/cfd/evaluation/datasets/adapter_registry.py` — base class and registry
- `physicsnemo/cfd/evaluation/datasets/schema.py` — `CanonicalCase` and `build_predictions_dict`
- `physicsnemo/cfd/evaluation/datasets/adapters/drivaerml.py` — reference adapter implementation
- `workflows/benchmarking_workflow/notebooks/adding_a_new_dataset.ipynb` — end-to-end tutorial

## Step 1: Explore the new dataset

Ask the user for the dataset path, then inspect one file:

```python
import pyvista as pv
mesh = pv.read("<path_to_one_file>")
print(f"Type: {type(mesh).__name__}, Points: {mesh.n_points}, Cells: {mesh.n_cells}")
print(f"Cell arrays: {list(mesh.cell_data.keys())}")
print(f"Point arrays: {list(mesh.point_data.keys())}")
```

Identify these differences from the canonical schema:

| Question | What to look for |
|----------|-----------------|
| File format | `.vtp`, `.vtu`, `.vtk`, or other? Model wrappers expect `.vtp` (surface) or `.vtu` (volume) XML format. |
| Directory layout | Flat directory? Nested `run_<id>/` dirs? How are case IDs derived from filenames? |
| Pressure field name | The canonical key is `pressure`. What is the VTK array name? |
| WSS field name | The canonical key is `shear_stress` (N, 3). Is it a single vector or separate scalar components? |
| Sign conventions | Compare field ranges with DrivAerML. Are normals, WSS, or pressure flipped? |
| Extra arrays | Are there explicit `Normals` or `Area` arrays? DrivAerML has none — remove them if present. |
| STL files | Are separate STL geometry files available? If not, the surface mesh itself is the geometry. |
| Inference domain | Surface (`.vtp`) or volume (`.vtu`)? |

## Step 2: Write the adapter class

Subclass `DatasetAdapter` with these methods:

```python
from pathlib import Path
from physicsnemo.cfd.evaluation.datasets.adapter_registry import DatasetAdapter, register_adapter
from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase

class MyDatasetAdapter(DatasetAdapter):
    def __init__(self, root: str, **kwargs):
        self._root = Path(root)

    @classmethod
    def inference_domain_from_kwargs(cls, kwargs=None):
        return "surface"  # or "volume"

    def list_cases(self, split=None):
        # Return list of case ID strings
        ...

    def load_case(self, case_id: str) -> CanonicalCase:
        # 1. Read the mesh file
        # 2. Build ground_truth dict with canonical keys:
        #    - "pressure": np.float32 array
        #    - "shear_stress": np.float32 array of shape (N, 3)
        #    For volume: "pressure", "velocity" (N,3), "turbulent_viscosity"
        # 3. Return CanonicalCase(case_id, mesh_path, mesh_type, ground_truth, inference_domain)
        ...
```

### Common transformations in `load_case`

**Format conversion** (legacy `.vtk` → `.vtp`):
```python
mesh = pv.read(vtk_path).extract_surface()
mesh.save(vtp_path)
```

**Combining separate WSS scalars into a vector:**
```python
wss = np.stack([mesh.cell_data["WSSx"], mesh.cell_data["WSSy"], mesh.cell_data["WSSz"]], axis=1)
```

**Removing explicit Normals/Area** (DrivAerML convention):
```python
for key in ["Normals", "Area"]:
    if key in mesh.cell_data:
        del mesh.cell_data[key]
```

**Creating STL from surface mesh** (when no STL is shipped):
```python
mesh.extract_surface().triangulate().save(stl_path)
```

The STL must be named `drivaer_{int(case_id)}.stl` in the same directory as the VTP for the model wrappers to find it.

### Caching pattern

Do expensive conversions lazily and cache:

```python
def _prepare_case(self, case_id):
    prepared_path = self._root / "_prepared" / f"{case_id}.vtp"
    if not prepared_path.exists():
        # ... convert and save
    return str(prepared_path)
```

## Step 3: Register and test

```python
register_adapter("my_dataset", MyDatasetAdapter)

adapter = MyDatasetAdapter(root="/path/to/data")
cases = adapter.list_cases()
case = adapter.load_case(cases[0])
assert case.ground_truth is not None
assert "pressure" in case.ground_truth
```

## Step 4: Run inference and benchmark

Build a config and run:

```python
from physicsnemo.cfd.evaluation.config import Config
from physicsnemo.cfd.evaluation.benchmarks.engine import run_benchmark

config = Config.from_dict({
    "run": {"device": "cuda:0", "output_dir": "results"},
    "model": {"name": "<model_name>", "inference_domain": "<surface|volume>", ...},
    "dataset": {"name": "my_dataset", "root": "/path/to/data", "case_ids": cases[:2]},
    "output": {
        "ground_truth_mesh_field_names": {"pressure": "<vtk_gt_name>", "shear_stress": "<vtk_gt_name>"},
        "mesh_field_names": {"pressure": "<vtk_pred_name>", "shear_stress": "<vtk_pred_name>"},
    },
    "metrics": ["l2_pressure", "l2_shear_stress", "drag", "lift"],
    "reports": {"enabled": False},
})
results = run_benchmark(config)
```

## Step 5: Make permanent (optional)

Save the adapter to `physicsnemo/cfd/evaluation/datasets/adapters/<name>.py` and register in `adapters/__init__.py`:

```python
from physicsnemo.cfd.evaluation.datasets.adapters.<name> import MyDatasetAdapter
register_adapter("my_dataset", MyDatasetAdapter)
```

## Why conventions must match the training data

The field name mappings, sign conventions, and format conversions in the adapter exist because the model checkpoint was trained on a specific dataset (e.g., DrivAerML) with specific conventions. The adapter bridges the gap between the new dataset's conventions and the training data's conventions — not some abstract standard. If a model is retrained directly on the new dataset, the adapter would not need these transformations. When writing an adapter, always ask: "What conventions did the model's training data use?" and map to those.

## Gotchas

- **DistributedManager**: Model wrappers call `DistributedManager.initialize()`. In notebooks without `torchrun`, set env vars first: `WORLD_SIZE=1`, `RANK=0`, `LOCAL_RANK=0`, `MASTER_ADDR=localhost`, `MASTER_PORT=12355`.
- **STL naming**: DoMINO looks for `drivaer_{tag}.stl`, GeoTransolver looks for `drivaer_{tag}_single_solid.stl` then `*.stl`. Both now fall back to any `*.stl` in the directory.
- **VTP vs VTK**: Model wrappers use VTK XML readers internally. Legacy `.vtk` files must be converted to `.vtp`/`.vtu`.
- **Checkpoint loading**: Some wrappers need `trusted_torch_load_context()` for PyTorch 2.6+ checkpoint compatibility.
- **Domain-scoped metrics**: `l2_pressure` resolves to different implementations for surface vs volume based on `inference_domain`. Use the same metric name for both.
