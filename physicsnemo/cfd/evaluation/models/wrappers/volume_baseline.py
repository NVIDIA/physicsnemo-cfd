# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Volume baseline wrapper: zeros (or shape-correct) predictions for pipeline tests."""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np
import pyvista as pv

from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    InferenceDomain,
    build_predictions_dict,
)
from physicsnemo.cfd.evaluation.models.model_registry import (
    CFDModel,
    ModelInput,
    OutputLocation,
    RawOutput,
    Predictions,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference


class VolumeBaselineWrapper(CFDModel):
    """No trained weights: zeros for ``pressure``, ``velocity``, and ``turbulent_viscosity`` on the volume mesh.

    ``OUTPUT_LOCATION`` is ``point`` so array lengths match :func:`~physicsnemo.cfd.evaluation.metrics.mesh_bridge.build_comparison_mesh`
    for volume (it attaches fields on mesh points, not cells).
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "volume"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "point"
    REQUIRES_REMOTE_ASSETS: ClassVar[bool] = False

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

    def decode_outputs(
        self,
        raw_output: RawOutput,
        case: CanonicalCase,
        model_input: Optional[ModelInput] = None,
    ) -> Predictions:
        log_inference(
            "volume_baseline",
            f"Reading volume mesh and building baseline fields: {case.mesh_path}",
        )
        mesh = pv.read(case.mesh_path)
        if hasattr(mesh, "cast_to_unstructured_grid"):
            mesh = mesh.cast_to_unstructured_grid()
        n = mesh.n_cells if self.output_location == "cell" else mesh.n_points
        p = np.zeros((n,), dtype=np.float32)
        v = np.zeros((n, 3), dtype=np.float32)
        nut = np.zeros((n,), dtype=np.float32)
        return build_predictions_dict(pressure=p, velocity=v, turbulent_viscosity=nut)
