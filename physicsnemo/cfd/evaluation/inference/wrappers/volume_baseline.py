"""Volume baseline wrapper: zeros (or shape-correct) predictions for pipeline tests."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pyvista as pv

from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    InferenceDomain,
    build_predictions_dict,
)
from physicsnemo.cfd.evaluation.inference.model_registry import (
    CFDModel,
    ModelInput,
    OutputLocation,
    RawOutput,
    Predictions,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference


class VolumeBaselineWrapper(CFDModel):
    """No trained weights: zeros for ``pressure_volume`` and ``turbulent_viscosity`` on the volume mesh."""

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "volume"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"

    @property
    def output_location(self) -> OutputLocation:
        return self.OUTPUT_LOCATION

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> VolumeBaselineWrapper:
        log_inference(
            "volume_baseline",
            "No checkpoint to load (baseline stub).",
        )
        return self

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        log_inference(
            "volume_baseline",
            f"Preparing inputs (case {case.case_id}; mesh read in decode step).",
        )
        return None

    def predict(self, model_input: ModelInput) -> RawOutput:
        log_inference("volume_baseline", "Running forward pass (no-op for baseline)…")
        return None

    def decode_outputs(self, raw_output: RawOutput, case: CanonicalCase) -> Predictions:
        log_inference(
            "volume_baseline",
            f"Reading volume mesh and building baseline fields: {case.mesh_path}",
        )
        mesh = pv.read(case.mesh_path)
        if hasattr(mesh, "cast_to_unstructured_grid"):
            mesh = mesh.cast_to_unstructured_grid()
        n = mesh.n_cells if self.output_location == "cell" else mesh.n_points
        return build_predictions_dict(
            pressure_volume=np.zeros((n,), dtype=np.float32),
            turbulent_viscosity=np.zeros((n,), dtype=np.float32),
        )
