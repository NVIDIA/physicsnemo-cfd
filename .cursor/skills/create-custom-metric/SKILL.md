---
name: create-custom-metric
description: >-
  Create a custom metric for the PhysicsNeMo CFD benchmarking workflow.
  Use when the user wants to add a new evaluation metric, implement a custom
  error measure, compute force coefficients, or extend the benchmark with
  domain-specific quantities.
---

# Create a Custom Metric

Guide the user through adding a new metric to the benchmarking workflow.

## Reference files to read first

- `physicsnemo/cfd/postprocessing_tools/metric_registry.py` — `register_metric`, `get_metric`, `MetricFn`
- `physicsnemo/cfd/evaluation/metrics/builtin/forces.py` — `drag_error`, `lift_error` (dict-returning, mesh-based)
- `physicsnemo/cfd/evaluation/metrics/builtin/l2.py` — L2 metrics (scalar-returning, numpy fallback)
- `physicsnemo/cfd/evaluation/metrics/mesh_bridge.py` — `build_comparison_mesh`
- `physicsnemo/cfd/postprocessing_tools/metrics/aero_forces.py` — `compute_force_coefficients` (normals, areas, integration)
- `workflows/benchmarking_workflow/notebooks/adding_a_new_metric.ipynb` — end-to-end tutorial

## Metric function signature

Metrics are plain callables, no base class:

```python
MetricFn = Callable[..., float | dict[str, float]]
```

**Modern signature** (accepts extended engine kwargs):

```python
def my_metric(
    ground_truth: dict,      # canonical GT: {"pressure": ..., "shear_stress": ...}
    predictions: dict,        # canonical predictions from decode_outputs
    *,
    case: Any = None,         # CanonicalCase from the dataset adapter
    comparison_mesh: Any = None,  # PyVista mesh with GT + pred arrays attached
    metric_dtype: str | None = None,  # "cell" or "point"
    output: Any = None,       # OutputConfig with field name mappings
    **_: object,              # absorb unknown kwargs
) -> float | dict[str, float]:
    ...
```

**Return types**:
- `float` — single scalar value (e.g., L2 error)
- `dict[str, float]` — multiple values; keys are auto-flattened by the engine: `{"error": 0.1, "pred": 42.0}` from metric `side_force` becomes `side_force_error` and `side_force_pred` in results

## Step 1: Write the metric function

### Simple array-based metric (no mesh needed)

```python
import numpy as np

def mae_pressure(ground_truth, predictions, **_):
    gt = np.asarray(ground_truth.get("pressure", []), dtype=np.float64).ravel()
    pred = np.asarray(predictions.get("pressure", []), dtype=np.float64).ravel()
    if gt.size == 0 or pred.size == 0 or gt.shape != pred.shape:
        return float("nan")
    return float(np.mean(np.abs(gt - pred)))
```

### Mesh-based metric (uses normals, areas, geometry)

Use `_resolve_mesh` pattern to get the comparison mesh, then access arrays:

```python
from physicsnemo.cfd.evaluation.metrics.mesh_bridge import build_comparison_mesh

def _resolve_mesh(predictions, *, case, comparison_mesh, metric_dtype, output):
    if comparison_mesh is not None and metric_dtype is not None:
        return comparison_mesh, metric_dtype
    if case is not None and output is not None:
        return build_comparison_mesh(case, predictions, output)
    return None, None

def my_force_metric(ground_truth, predictions, *, case=None, comparison_mesh=None,
                    metric_dtype=None, output=None, **_):
    mesh, dtype = _resolve_mesh(predictions, case=case, comparison_mesh=comparison_mesh,
                                 metric_dtype=metric_dtype, output=output)
    if mesh is None or output is None:
        return float("nan")

    # Access fields by VTK array name from output config
    p = mesh.cell_data[output.mesh_field_names["pressure"]]
    wss = mesh.cell_data[output.mesh_field_names["shear_stress"]]

    # Access mesh geometry
    mesh = mesh.compute_normals().compute_cell_sizes()
    normals = mesh["Normals"]   # (N, 3)
    areas = mesh["Area"]        # (N,)

    # Compute your metric...
    return float(result)
```

## Step 2: Register the metric

```python
from physicsnemo.cfd.postprocessing_tools.metric_registry import register_metric

register_metric("my_metric", my_metric_fn, domain="surface")  # or "volume" or None
```

- `domain="surface"` — only used when model's inference domain is surface
- `domain="volume"` — only used for volume inference
- `domain=None` — domain-agnostic fallback
- Same name can be registered for both domains with different functions (like `l2_pressure`)

## Step 3: Use in benchmark config

Add the metric name to the `metrics` list:

```python
config = Config.from_dict({
    ...
    "metrics": ["l2_pressure", "drag", "lift", "my_metric"],
    ...
})
```

Or in YAML:
```yaml
metrics:
  - l2_pressure
  - my_metric
```

Per-metric kwargs can be passed as a dict:
```yaml
metrics:
  - name: my_metric
    some_param: 42
```

## Step 4: Make permanent (optional)

Add to `physicsnemo/cfd/evaluation/metrics/builtin/` and register from `builtin/__init__.py`:

```python
def register_my_metrics():
    register_metric("my_metric", my_fn, domain="surface")

# In __init__.py:
def register_all_builtin_metrics():
    register_l2_metrics()
    register_force_metrics()
    register_physics_metrics()
    register_my_metrics()  # add this
```

## Existing built-in metrics

| Name | Domain(s) | Returns |
|------|-----------|---------|
| `l2_pressure` | surface, volume | `float` |
| `l2_shear_stress` | surface | `dict` |
| `l2_pressure_area_weighted` | surface | `float` |
| `l2_velocity` | volume | `dict` |
| `l2_turbulent_viscosity` | volume | `float` |
| `drag` | surface | `dict` (error, true, pred) |
| `lift` | surface | `dict` (error, true, pred) |
| `continuity_residual_l2` | volume | `float` |
| `momentum_residual_l2` | volume | `float` |

## Gotchas

- **Dict flattening**: if metric returns `{"error": 0.1, "true": 5.0}`, engine stores as `metricname_error` and `metricname_true`. An empty string key `""` maps to just `metricname`.
- **NaN handling**: return `float("nan")` for failures; engine accumulates NaN gracefully.
- **Legacy fallback**: engine tries extended kwargs first; on `TypeError` it falls back to `fn(gt, predictions, **mkwargs)` only. Modern metrics should accept `**_` to absorb unknowns.
- **Results JSON format**: `benchmark_results.json` is a plain `list[dict]`, not `{"results": [...]}`.
- **OutputConfig field names**: surface uses `output.mesh_field_names` / `output.ground_truth_mesh_field_names`; volume uses `output.volume_mesh_field_names` / `output.ground_truth_volume_mesh_field_names`.
