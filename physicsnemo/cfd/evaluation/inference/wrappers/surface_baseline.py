"""Surface baseline wrapper: zeros on boundary VTP for pipeline / smoke tests."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from physicsnemo.cfd.evaluation.common.io import load_mesh
from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    InferenceDomain,
    predictions_dict,
)
from physicsnemo.cfd.evaluation.inference.model_registry import (
    CFDModel,
    ModelInput,
    OutputLocation,
    RawOutput,
    Predictions,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference


class SurfaceBaselineWrapper(CFDModel):
    """No trained weights: zeros for ``pressure`` and ``shear_stress`` on the surface mesh.

    Uses **cell**-centered counts (same as many surface wrappers here). Pair with
    ``drivaerml`` default surface branch and ``align_ground_truth_to_model`` / GT
    location as needed for metrics.
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"
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
    ) -> SurfaceBaselineWrapper:
        log_inference(
            "surface_baseline",
            "No checkpoint to load (baseline stub).",
        )
        return self

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        log_inference(
            "surface_baseline",
            f"Preparing inputs (case {case.case_id}; mesh read in decode step).",
        )
        return None

    def predict(self, model_input: ModelInput) -> RawOutput:
        log_inference("surface_baseline", "Running forward pass (no-op for baseline)…")
        return None

    def decode_outputs(self, raw_output: RawOutput, case: CanonicalCase) -> Predictions:
        log_inference(
            "surface_baseline",
            f"Reading surface mesh and building baseline fields: {case.mesh_path}",
        )
        mesh = load_mesh(case.mesh_path)
        n = mesh.n_cells if self.output_location == "cell" else mesh.n_points
        p = np.zeros((n,), dtype=np.float32)
        wss = np.zeros((n, 3), dtype=np.float32)
        return predictions_dict(p, wss)
