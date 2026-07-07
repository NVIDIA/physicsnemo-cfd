---
name: physicsnemo-cfd-create-model-wrapper
description: >-
  Create a new model wrapper for the PhysicsNeMo CFD benchmarking workflow.
  Use when the user wants to add a new CFD model, write a CFDModel wrapper,
  integrate a new neural network architecture, or run a custom model through
  the benchmarking pipeline.
license: Apache-2.0
---

# Create a Model Wrapper

Guide the user through adding a new CFD model to the benchmarking workflow
by writing a `CFDModel` subclass.

## First: gather whatever context already exists

Spend a little effort up front collecting any existing artifacts that
reveal how the model actually behaves — but **do not block on them**.
Look (without stopping to ask the user) for:

- the **training/inference script** and any
  preprocessing/normalization utilities it imports;
- the model's **config files** (YAML/JSON hyperparameters, channel lists,
  stats paths);
- the model class's **docstrings and signatures** (forward inputs/outputs,
  expected shapes).

Use these to pin down four things the wrapper must mirror exactly:

- **Normalization** — which scheme and which stats (see Normalization
  below).
- **Inputs** — what the model actually consumes (coordinates only, extra
  fields, geometry/STL).
- **Output fields and order** — which variables the forward pass returns
  and in what channel order.
- **I/O shapes and dtypes** — so `prepare_inputs`/`decode_outputs` match
  the trained graph.

If some or all of these aren't available, that's fine — **proceed
anyway**: reconstruct from the checkpoint and stats file, pick the most
likely option, and build the wrapper now. Do **not** stop and wait for
answers before writing code. Just avoid *silently* guessing: a wrong
normalization scheme or channel order produces plausible-looking but
wrong predictions. So state each such assumption inline, and after
delivering the wrapper, **raise the still-uncertain choices as explicit
open items for the user to confirm** — e.g. normalization scheme, input
tier, and output fields/channel order. This keeps you unblocked while
making the risky decisions visible.

## Reference files to read first

Start with the complete, ready-to-adapt templates bundled with this skill
— they are always available even when the PhysicsNeMo-CFD source tree is
not on disk:

- `references/example_wrapper.py` — full surface **and** volume
  `CFDModel` reference implementations (load, prepare_inputs, predict,
  decode_outputs, registration). Copy and adapt one of these.
- `assets/global_stats.example.json` — sample mean/std stats for both
  surface and volume.

When the PhysicsNeMo-CFD repo *is* present, also read these for the live
interface (verify paths against the actual tree):

- `physicsnemo/cfd/evaluation/models/model_registry.py` — base class and
  registry
- `physicsnemo/cfd/evaluation/datasets/schema.py` — `CanonicalCase`,
  `build_predictions_dict`
- `physicsnemo/cfd/evaluation/models/wrappers/surface_baseline.py` —
  simplest concrete surface wrapper
- `physicsnemo/cfd/evaluation/models/wrappers/volume_baseline.py` —
  simplest concrete volume wrapper
- `physicsnemo/cfd/evaluation/models/wrappers/__init__.py` — how wrappers
  are registered
- `physicsnemo/cfd/evaluation/common/io.py` — mesh loading and
  normalization stats helpers
- `workflows/benchmarking/notebooks/adding_a_new_model.ipynb` —
  end-to-end tutorial

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

The engine calls `load` once, then `prepare_inputs → predict →
decode_outputs(raw, case, model_input)` per case (`model_input` is the
dict from `prepare_inputs`; use when decode must mirror inference
geometry).

## Step 1: Write the wrapper class

**Always generate a new, complete wrapper class for the requested
model.** Existing wrappers (e.g. `surface_baseline.py`) are *references
to read*, not substitutes — when the user asks to write or create a
wrapper, produce a full new class file even if similar ones already
exist. Do not stop at "a wrapper already exists" or offer to reuse one
in place of writing the requested one. Always tell the user how to
register it: `register_model(...)` at import for a quick test, and an
entry in `wrappers/__init__.py` to make it permanent (Step 6).

### Anti-patterns (do not do these)

- **Reusing an existing wrapper instead of writing the requested one** —
  even if `git status` shows a similar file, write the new class.
- **Assuming mean-std normalization** — confirm the scheme; a wrong
  inverse gives plausible-but-wrong fields.
- **Hardcoding `pressure` + `shear_stress`** — pass only the fields the
  model predicts; add custom ones via `**extra`.
- **Skipping `__init__.py`** — registering only inline and never
  mentioning permanent registration.
- **Echoing the interface table or full reference file back** — wastes
  tokens; reference, don't repeat.

**Copy `references/example_wrapper.py` and adapt it** — it has full
surface and volume implementations. Don't hand-write from scratch or
paste the whole template back to the user. The class skeleton is just:

```python
class MyModelWrapper(CFDModel):
    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"  # or "volume"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"        # or "point"

    @property
    def output_location(self): return self.OUTPUT_LOCATION
    def load(self, checkpoint_path, stats_path, device, **kwargs): ...   # weights + stats; return self
    def prepare_inputs(self, case): ...                                  # CanonicalCase -> model input
    def predict(self, model_input): ...                                  # forward pass -> raw output
    def decode_outputs(self, raw_output, case, model_input=None): ...    # denormalize -> build_predictions_dict(...)
```

Keep responses terse: state the few model-specific decisions
(normalization scheme, input tier, output fields) and the file you
wrote — don't echo the interface table or the full reference file back.

### Key implementation considerations

**Normalization** (match the training script exactly): Most trained
models normalize inputs/outputs, and `decode_outputs` must apply the
*inverse* of whatever the model was trained with. First identify the
scheme:

- **Mean-std (z-score)**: `x_norm = (x - mean) / std` → inverse
  `x = x_norm * std + mean`. This is the repo's built-in format. Use
  `load_global_stats(stats_path)` from
  `physicsnemo/cfd/evaluation/common/io.py`; it reads `mean`/`std_dev`
  JSON and returns `mean`/`std` tensors.
- **Min-max**: `x_norm = (x - min) / (max - min)` → inverse
  `x = x_norm * (max - min) + min`. There is **no built-in helper** for
  this — store `min`/`max` (e.g. in your stats JSON) and apply the
  inverse yourself in `decode_outputs`. Do not feed a min-max file to
  `load_global_stats`; the keys won't match.

Confirm the scheme from the training/inference script or the stats file
rather than assuming mean-std. Applying the wrong inverse yields
wrong-but-plausible fields that still pass shape checks.

**Inputs** (handle the model's actual input tier): `prepare_inputs`
receives a `CanonicalCase`. Pull what the model needs:

- **Point cloud only**: coordinates from `case.mesh_path` (vtp/vtu) via
  `pv.read`, as in the example — sufficient for many geometry-only
  models.
- **Extra field inputs** (e.g. inlet/freestream velocity, Reynolds
  number): read from `case.metadata` (or `case.ground_truth` for field
  arrays). Broadcast/concatenate them onto the per-point features as the
  training script did.
- **Geometry/STL** (e.g. SDF or BVH-based models): the STL/geometry path
  is typically on `case.metadata`; load it in `prepare_inputs` and build
  the geometric encoding the model expects.

Inspect `case.metadata` and `case.ground_truth` keys for a real case
early — the dataset adapter decides what is available.

**Outputs** (predict only what the model produces, plus extras):
`build_predictions_dict` takes `pressure`, `shear_stress`, `velocity`,
`turbulent_viscosity` (all optional) **and arbitrary `**extra`
fields**. So:

- A model that predicts only WSS magnitude, or no WSS at all, simply
  omits the missing keys — pass only what it produces.
- Extra/non-standard outputs (e.g. `stagnation_pressure`, `temperature`,
  `mach`) are passed as keyword args: `build_predictions_dict(pressure=p,
  mach=m, temperature=t)`. Each becomes a prediction variable.
- For a custom field to appear in the written mesh and metrics, add a
  matching entry to `output.mesh_field_names` in the config (Step 4) and
  a corresponding metric if you want it scored.

**Output shape**: `pressure` must be `(N,)` float32. `shear_stress` must
be `(N, 3)` float32 for surface. Volume fields: `velocity` is `(N, 3)`,
`turbulent_viscosity` is `(N,)`. Custom scalar fields are `(N,)`, vector
fields `(N, k)`.

**Output location**: If `OUTPUT_LOCATION = "cell"`, return N =
`mesh.n_cells` values. If `"point"`, return N = `mesh.n_points` values.

**Batching**: For large meshes, `prepare_inputs` may need to subsample
or batch. Use `kwargs` passed through `load()` (e.g.,
`batch_resolution`, `geometry_sampling`) to control this.

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

This `mean`/`std_dev` layout is what `load_global_stats()` expects
(mean-std models). If your model was trained with **min-max**
normalization, this helper does not apply — persist `min`/`max` per
field in your own JSON and apply the inverse manually in
`decode_outputs` (see Normalization above).

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

Results are written to `benchmark_results.json` (a JSON list of dicts,
one per model×dataset combo).

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

Save the wrapper to
`physicsnemo/cfd/evaluation/models/wrappers/my_model.py` and register in
`wrappers/__init__.py`:

```python
from physicsnemo.cfd.evaluation.models.wrappers.my_model import MyModelWrapper
register_model("my_model", MyModelWrapper)
```

Then use `model.name: my_model` in any YAML config.

## Gotchas

- **DistributedManager**: Model wrappers may call
  `DistributedManager.initialize()`. In notebooks without `torchrun`,
  set env vars first: `WORLD_SIZE=1`, `RANK=0`, `LOCAL_RANK=0`,
  `MASTER_ADDR=localhost`, `MASTER_PORT=12355`.
- **`weights_only=True`**: Use this flag with `torch.load()` for safe
  deserialization (PyTorch 2.6+ default).
- **Domain matching**: The engine checks that `model.INFERENCE_DOMAIN`
  matches the dataset adapter's `inference_domain_from_kwargs()`.
  Mismatches are skipped in matrix mode or raise in single mode.
- **GT alignment**: When `align_ground_truth_to_model: true` in dataset
  kwargs, the engine converts GT data to match `OUTPUT_LOCATION` (point
  ↔ cell). This is automatic — the wrapper just needs correct class
  vars.
- **Results JSON format**: `benchmark_results.json` is a plain
  `list[dict]`, not `{"results": [...]}`. Iterate directly:
  `for combo in report:`.

## Related resources

- `references/example_wrapper.py` — complete surface + volume `CFDModel`
  templates to copy and adapt (bundled; available without the repo on
  disk).
- `assets/global_stats.example.json` — sample mean/std stats layout for
  surface and volume.
