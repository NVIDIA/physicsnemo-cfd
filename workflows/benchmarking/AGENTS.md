# Agent Instructions for Benchmarking Workflow

This file provides guidance for AI coding agents working on the PhysicsNeMo CFD benchmarking workflow.

## Available Skills

### Creating a New Dataset Adapter

When the user wants to add a new CFD dataset, benchmark models on custom data, or integrate a new mesh format, follow the instructions in [`.cursor/skills/create-dataset-adapter/SKILL.md`](../../.cursor/skills/create-dataset-adapter/SKILL.md).

Key reference files:
- `physicsnemo/cfd/evaluation/datasets/adapter_registry.py` — `DatasetAdapter` base class
- `physicsnemo/cfd/evaluation/datasets/adapters/drivaerml.py` — reference implementation
- `workflows/benchmarking/notebooks/adding_a_new_dataset.ipynb` — end-to-end tutorial

### Creating a New Model Wrapper

When the user wants to add a new CFD model, integrate a new neural network architecture, or run a custom model through the benchmarking pipeline, follow the instructions in [`.cursor/skills/create-model-wrapper/SKILL.md`](../../.cursor/skills/create-model-wrapper/SKILL.md).

Key reference files:
- `physicsnemo/cfd/evaluation/models/model_registry.py` — `CFDModel` base class
- `physicsnemo/cfd/evaluation/models/wrappers/surface_baseline.py` — simplest reference implementation
- `workflows/benchmarking/notebooks/adding_a_new_model.ipynb` — end-to-end tutorial

### Creating a Custom Metric

When the user wants to add a new evaluation metric, implement a custom error measure, compute force coefficients, or extend the benchmark with domain-specific quantities, follow the instructions in [`.cursor/skills/create-custom-metric/SKILL.md`](../../.cursor/skills/create-custom-metric/SKILL.md).

Key reference files:
- `physicsnemo/cfd/postprocessing_tools/metric_registry.py` — `register_metric`, `MetricFn`
- `physicsnemo/cfd/evaluation/metrics/builtin/forces.py` — reference force metric implementation
- `workflows/benchmarking/notebooks/adding_a_new_metric.ipynb` — end-to-end tutorial

## Codebase Conventions

- **Canonical keys**: Surface fields use `pressure` and `shear_stress`. Volume fields use `pressure`, `velocity`, `turbulent_viscosity`. The same key `pressure` is used for both domains — the metric registry resolves the correct implementation via `domain` scoping.
- **Metric names**: Use `l2_pressure`, `l2_shear_stress`, `drag`, `lift` etc. in configs. Metrics are domain-scoped — the engine passes `inference_domain` automatically.
- **Config**: Hydra YAML configs under `conf/`. Programmatic configs via `Config.from_dict()`.
- **Model wrappers**: Registered in `physicsnemo/cfd/evaluation/models/wrappers/`. Each implements `load`, `prepare_inputs`, `predict`, `decode_outputs`.
- **Visual plugins**: Registered in `physicsnemo/cfd/evaluation/reports/builtin/`. Each follows `(config, results, output_dir, *, context, **kwargs) -> None`.
