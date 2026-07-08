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

"""Self-contained reference wrappers for the PhysicsNeMo CFD benchmarking workflow.

These are complete, correct ``CFDModel`` implementations to **adapt** when writing a
new wrapper — one surface model and one volume model. They are built from the real
evaluation APIs and the patterns in the shipped baseline wrappers
(``physicsnemo/cfd/evaluation/models/wrappers/surface_baseline.py`` and
``volume_baseline.py``) plus the ``adding_a_new_model.ipynb`` tutorial, so they stay
usable as a template even when the full PhysicsNeMo source tree is not on disk. When the
repo is present, verify the imported names against it — these templates are hints, not
ground truth.

Adapt, don't copy blindly. The four things every wrapper must mirror from the model's
own training/inference code are flagged inline below:
  (1) NORMALIZATION scheme, (2) INPUT tier, (3) OUTPUT fields + channel order, (4) shapes.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np
import torch
import pyvista as pv

from physicsnemo.cfd.evaluation.common.io import (
    load_global_stats,
    volume_dataset_from_case,
)
from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    InferenceDomain,
    build_predictions_dict,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference
from physicsnemo.cfd.evaluation.models.model_registry import (
    CFDModel,
    OutputLocation,
    register_model,
)


class ExampleSurfaceWrapper(CFDModel):
    """Minimal surface wrapper: predicts ``pressure`` + ``shear_stress`` on a surface mesh.

    The simplest end-to-end contract: read cell-center coordinates, run a forward pass,
    return canonical predictions. Use this shape for geometry-only surface models.
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._stats: dict[str, Any] | None = None
        self._device: str = "cpu"

    @property
    def output_location(self) -> OutputLocation:
        """Whether predictions live on mesh cells or points ("cell" here)."""
        return self.OUTPUT_LOCATION

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "ExampleSurfaceWrapper":
        """Build the architecture, load weights, and load normalization stats.

        Returns ``self`` so the loaded model can be used fluently.
        """
        self._device = device
        # Build your architecture and load weights here, e.g.:
        #   self._model = MyNet(**kwargs.get("model_kwargs", {}))
        #   self._model.load_state_dict(
        #       torch.load(checkpoint_path, map_location=device, weights_only=True)
        #   )
        #   self._model.to(device).eval()
        # (1) NORMALIZATION: load the SAME stats the model trained with. load_global_stats
        # handles the mean/std_dev JSON format; for min-max, load your own min/max here.
        self._stats = load_global_stats(stats_path, device)
        log_inference("example_surface", f"Loaded from {checkpoint_path}")
        return self

    def prepare_inputs(self, case: CanonicalCase) -> torch.Tensor:
        """Turn a canonical case into the tensor the model's forward pass expects."""
        # (2) INPUT tier: this model consumes coordinates only. For models that need
        # extra fields (inlet velocity, Re) or geometry, pull them from case.metadata /
        # case.ground_truth and concatenate/encode them here.
        mesh = pv.read(case.mesh_path)
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()
        coords = np.array(mesh.cell_centers().points, dtype=np.float32)
        return torch.tensor(coords, device=self._device)

    def predict(self, model_input: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the forward pass and return raw (still-normalized) field tensors."""
        with torch.no_grad():
            # raw = self._model(model_input)
            # Placeholder so the template runs as a smoke test:
            n = model_input.shape[0]
            raw = {
                "pressure": torch.zeros(n, device=self._device),
                "shear_stress": torch.zeros((n, 3), device=self._device),
            }
        return raw

    def decode_outputs(
        self,
        raw_output: dict[str, torch.Tensor],
        case: CanonicalCase,
        model_input: Optional[torch.Tensor] = None,
    ) -> dict[str, np.ndarray]:
        """Denormalize raw outputs and pack them into a canonical predictions dict."""
        # (1) NORMALIZATION: apply the inverse of training normalization before returning.
        # (3)/(4) OUTPUT: pressure is (N,), shear_stress is (N, 3). Pass only the fields
        # this model predicts; omit any it does not (e.g. drop shear_stress for a
        # pressure-only model). Custom fields are allowed as extra kwargs, e.g.
        #   build_predictions_dict(pressure=p, mach=m, temperature=t)
        return build_predictions_dict(
            pressure=raw_output["pressure"].cpu().numpy(),
            shear_stress=raw_output["shear_stress"].cpu().numpy(),
        )


class ExampleVolumeWrapper(CFDModel):
    """Volume wrapper: loads a ``.pt`` checkpoint + ``global_stats.json`` and denormalizes.

    Shows the full real-model path: build architecture from kwargs, load weights and
    stats, handle output channel order, and denormalize every volume field.
    Output channels (5 total): velocity (3), pressure (1), turbulent_viscosity (1).
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "volume"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "point"

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._stats: dict[str, Any] | None = None
        self._device: str = "cpu"

    @property
    def output_location(self) -> OutputLocation:
        """Whether predictions live on mesh cells or points ("point" here)."""
        return self.OUTPUT_LOCATION

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "ExampleVolumeWrapper":
        """Load the checkpoint weights and normalization stats onto ``device``.

        Returns ``self`` so the loaded model can be used fluently.
        """
        self._device = device
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Many training scripts nest the weights under a key — unwrap if present.
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # model = build_my_model(**kwargs.get("model_kwargs", {}))
        # model.load_state_dict(state_dict)
        # model.to(device).eval()
        # self._model = model

        # (1) NORMALIZATION: mean/std_dev stats, applied as inverse in decode_outputs.
        self._stats = load_global_stats(stats_path, device)
        log_inference("example_volume", f"Loaded checkpoint from {checkpoint_path}")
        return self

    def prepare_inputs(self, case: CanonicalCase) -> dict[str, Any]:
        """Extract volume points, optionally normalize coords, and bundle model inputs."""
        # (2) INPUT tier: volumetric point cloud. Normalize input coords if the model
        # was trained on normalized coordinates.
        mesh = volume_dataset_from_case(case)
        coords = np.array(mesh.points, dtype=np.float32)
        coords_t = torch.tensor(coords, device=self._device)
        if self._stats is not None and "coords" in self._stats["mean"]:
            mean = self._stats["mean"]["coords"]
            std = self._stats["std"]["coords"]
            std = torch.where(std.abs() < 1e-8, torch.ones_like(std), std)
            coords_t = (coords_t - mean) / std
        return {"coords": coords_t, "mesh": mesh}

    def predict(self, model_input: dict[str, Any]) -> torch.Tensor:
        """Run the forward pass and return the raw (N, 5) output tensor."""
        with torch.no_grad():
            # return self._model(model_input["coords"])
            n = model_input["coords"].shape[0]
            return torch.zeros((n, 5), device=self._device)

    def decode_outputs(
        self,
        raw_output: torch.Tensor,
        case: CanonicalCase,
        model_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        """Split output channels, denormalize each field, and build the predictions dict."""
        # (3) OUTPUT channel order MUST match training: velocity(3), pressure(1), nut(1).
        vel = raw_output[:, 0:3]
        pres = raw_output[:, 3:4].squeeze(-1)
        nut = raw_output[:, 4:5].squeeze(-1)

        # (1) NORMALIZATION: inverse mean/std per field.
        vel = self._denormalize(vel, "velocity")
        pres = self._denormalize(pres, "pressure")
        nut = self._denormalize(nut, "turbulent_viscosity")

        # (4) SHAPES: velocity (N, 3); pressure and turbulent_viscosity (N,).
        return build_predictions_dict(
            velocity=vel.cpu().numpy().reshape(-1, 3),
            pressure=pres.cpu().numpy().ravel(),
            turbulent_viscosity=nut.cpu().numpy().ravel(),
        )

    def _denormalize(self, tensor: torch.Tensor, field: str) -> torch.Tensor:
        if self._stats is None:
            return tensor
        mean = self._stats["mean"].get(field)
        std = self._stats["std"].get(field)
        if mean is None or std is None:
            return tensor
        std = torch.where(std.abs() < 1e-8, torch.ones_like(std), std)
        return tensor * std + mean


# Registration: do this once at import time so the engine can resolve the name.
register_model("example_surface", ExampleSurfaceWrapper)
register_model("example_volume", ExampleVolumeWrapper)
