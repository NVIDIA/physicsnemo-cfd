# Agent Instructions for Benchmarking Workflow

This file provides guidance for AI coding agents working on the PhysicsNeMo CFD benchmarking workflow.

## Available Skills

### Creating a New Dataset Adapter

When the user wants to add a new CFD dataset, benchmark models on custom data, or integrate a new mesh format, follow the instructions in [`.cursor/skills/create-dataset-adapter/SKILL.md`](../../.cursor/skills/create-dataset-adapter/SKILL.md).

Key reference files:
- `physicsnemo/cfd/evaluation/datasets/adapter_registry.py` — `DatasetAdapter` base class
- `physicsnemo/cfd/evaluation/datasets/adapters/drivaerml.py` — reference implementation
- `workflows/benchmarking_workflow/notebooks/adding_a_new_dataset.ipynb` — end-to-end tutorial

## Codebase Conventions

- **Canonical keys**: Surface fields use `pressure` and `shear_stress`. Volume fields use `pressure`, `velocity`, `turbulent_viscosity`. The same key `pressure` is used for both domains — the metric registry resolves the correct implementation via `domain` scoping.
- **Metric names**: Use `l2_pressure`, `l2_shear_stress`, `drag`, `lift` etc. in configs. Metrics are domain-scoped — the engine passes `inference_domain` automatically.
- **Config**: Hydra YAML configs under `conf/`. Programmatic configs via `Config.from_dict()`.
- **Model wrappers**: Registered in `physicsnemo/cfd/evaluation/inference/wrappers/`. Each implements `load`, `prepare_inputs`, `predict`, `decode_outputs`.
- **Visual plugins**: Registered in `physicsnemo/cfd/evaluation/reports/builtin/`. Each follows `(config, results, output_dir, *, context, **kwargs) -> None`.
