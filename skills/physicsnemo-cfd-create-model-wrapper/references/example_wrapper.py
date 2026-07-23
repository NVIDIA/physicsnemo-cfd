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
new wrapper. They cover:
  * ``ExampleSurfaceWrapper`` / ``ExampleVolumeWrapper`` — deterministic models.
  * ``ExampleAnalyticUQWrapper`` — a single-pass UQ model (GP head / mean-variance /
    evidential): ``UQ_METHOD="analytic"``, returns a predictive distribution directly.
  * ``ExampleSamplingUQWrapper`` — a multi-pass UQ model (MC-Dropout / deep ensemble):
    ``UQ_METHOD="sampling"``, the engine drives the passes and aggregates them.

They are built from the real evaluation APIs and the patterns in the shipped baseline and
UQ wrappers, so they stay usable as a template even when the full PhysicsNeMo source tree
is not on disk. When the repo is present, verify the imported names against it — these
templates are hints, not ground truth.

Adapt, don't copy blindly. The four things every wrapper must mirror from the model's
own training/inference code are flagged inline below:
  (1) NORMALIZATION scheme, (2) INPUT tier, (3) OUTPUT fields + channel order, (4) shapes.
For UQ wrappers there is a fifth: (5) the UQ CONTRACT (see the two UQ examples and the
"Uncertainty quantification" section of SKILL.md).
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterable, Optional

import numpy as np
import torch
import pyvista as pv

from physicsnemo.cfd.evaluation.common.io import (
    load_global_stats,
    volume_dataset_from_case,
)
from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    FieldDistribution,
    InferenceDomain,
    build_predictions_dict,
    build_predictive_distribution,
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


class ExampleAnalyticUQWrapper(CFDModel):
    """Single-pass UQ model (GP head / mean-variance / evidential): ``UQ_METHOD="analytic"``.

    The model itself emits the predictive distribution (or its parameters) in ONE forward
    pass. The extra contract vs a deterministic wrapper is:
      * declare ``SUPPORTS_UQ = True`` and ``UQ_METHOD = "analytic"``;
      * implement ``decode_distribution`` to return a ``FieldDistribution`` per field.
    ``decode_outputs`` is still provided (the distribution *mean*) so the deterministic
    metrics (L2 / drag / lift) and non-UQ runs keep working unchanged.
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"
    # (5) UQ CONTRACT: one forward pass -> a distribution, materialized in decode_distribution.
    SUPPORTS_UQ: ClassVar[bool] = True
    UQ_METHOD: ClassVar[str] = "analytic"

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._stats: dict[str, Any] | None = None
        self._device: str = "cpu"

    @property
    def output_location(self) -> OutputLocation:
        """Where predictions live (cell-centered surface, here)."""
        return self.OUTPUT_LOCATION

    def load(
        self, checkpoint_path: str, stats_path: str, device: str, **kwargs: Any
    ) -> "ExampleAnalyticUQWrapper":
        """Build the architecture + UQ head, load weights and normalization stats."""
        self._device = device
        # Build the architecture + UQ head and load weights here (e.g. a GP head whose
        # hyperparameters come from kwargs and MUST match training so the state dict loads).
        self._stats = load_global_stats(stats_path, device)
        log_inference("example_analytic_uq", f"Loaded from {checkpoint_path}")
        return self

    def prepare_inputs(self, case: CanonicalCase) -> torch.Tensor:
        """Read the case mesh and return the model-ready input tensor."""
        mesh = pv.read(case.mesh_path)
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()
        coords = np.array(mesh.cell_centers().points, dtype=np.float32)
        return torch.tensor(coords, device=self._device)

    def predict(self, model_input: torch.Tensor) -> dict[str, torch.Tensor]:
        """One forward pass -> raw posterior summary (still normalized).

        Return whatever the head produces; here the Gaussian summary mean + total std +
        the epistemic (model) component. A GP splits posterior variance (epistemic) from the
        learned noise floor (aleatoric); a mean-variance net returns mean + a variance head.
        """
        with torch.no_grad():
            n = model_input.shape[0]
            # raw = self._model(model_input)  # -> mean, variance, epistemic_variance, ...
            raw = {
                "mean": torch.zeros((n, 4), device=self._device),  # p + WSS(3)
                "total_std": torch.ones((n, 4), device=self._device),
                "epistemic_std": torch.ones((n, 4), device=self._device),
            }
        return raw

    def _to_physical(
        self, arr: torch.Tensor, field: str, *, is_std: bool
    ) -> np.ndarray:
        """Inverse mean-std. NOTE: means map with +mean; std/variance channels do NOT
        add the offset (they scale by ``std`` only). Denormalize BOTH here so metrics never
        touch normalization stats (all UQ metrics run in physical units)."""
        if self._stats is None:
            return arr.cpu().numpy()
        mean = self._stats["mean"].get(field, 0.0)
        std = self._stats["std"].get(field, 1.0)
        out = arr * std if is_std else arr * std + mean
        return out.cpu().numpy()

    def decode_distribution(
        self,
        raw_output: dict[str, torch.Tensor],
        case: CanonicalCase,
        model_input: Optional[torch.Tensor] = None,
    ) -> dict[str, FieldDistribution]:
        """Map the raw posterior to a ``FieldDistribution`` per canonical field (physical units).

        ``build_predictive_distribution`` mirrors ``build_predictions_dict`` but carries the
        std channels. Provide ``std`` (total) and, when the method separates them,
        ``epistemic_std`` / ``aleatoric_std`` — the pooled UQ metrics consume them directly.
        """
        mean = raw_output["mean"]
        tot = raw_output["total_std"]
        epi = raw_output["epistemic_std"]
        return {
            "pressure": build_predictive_distribution(
                mean=self._to_physical(mean[:, 0], "pressure", is_std=False),
                std=self._to_physical(tot[:, 0], "pressure", is_std=True),
                epistemic_std=self._to_physical(epi[:, 0], "pressure", is_std=True),
            ),
            "shear_stress": build_predictive_distribution(
                mean=self._to_physical(mean[:, 1:4], "shear_stress", is_std=False),
                std=self._to_physical(tot[:, 1:4], "shear_stress", is_std=True),
                epistemic_std=self._to_physical(
                    epi[:, 1:4], "shear_stress", is_std=True
                ),
            ),
        }

    def decode_outputs(
        self,
        raw_output: dict[str, torch.Tensor],
        case: CanonicalCase,
        model_input: Optional[torch.Tensor] = None,
    ) -> dict[str, np.ndarray]:
        """Point estimate (distribution mean) for the deterministic metric paths / non-UQ runs."""
        mean = raw_output["mean"]
        return build_predictions_dict(
            pressure=self._to_physical(mean[:, 0], "pressure", is_std=False),
            shear_stress=self._to_physical(mean[:, 1:4], "shear_stress", is_std=False),
        )


class ExampleSamplingUQWrapper(CFDModel):
    """Multi-pass UQ model (MC-Dropout / deep ensemble): ``UQ_METHOD="sampling"``.

    The wrapper only produces ONE (stochastic) pass at a time; the ENGINE drives ``N``
    passes and aggregates the spread into a ``FieldDistribution`` (streaming Welford). So the
    contract is:
      * declare ``SUPPORTS_UQ = True`` and ``UQ_METHOD = "sampling"``;
      * make ``predict`` produce a *different* draw each call (dropout stays stochastic at
        inference, or a weight sample is drawn) — the engine re-seeds per pass for
        reproducibility, so do NOT freeze the RNG yourself;
      * ``decode_outputs`` maps one raw pass to physical predictions (the engine calls it
        once per pass, then aggregates);
      * override ``predict_deterministic`` when ``predict`` is stochastic, so ``run.uq.enabled=
        false`` yields a true point prediction (MC-Dropout disables dropout for one pass; an
        ensemble returns a single member);
      * OPTIONAL multi-pass path: implement ``predict_ensemble(model_input, n)`` to yield the
        passes/members as an ``Iterable[RawOutput]`` (prefer a lazy generator so only one output
        is device-resident at a time). Honor ``n``: yield ``n`` passes for a per-model sampler, or
        ``min(n, member_count)`` members for a fixed-size ensemble. No ``decode_distribution`` is
        needed — the engine builds the distribution.

    The number of passes is the benchmark-wide ``run.uq.num_samples`` (NOT a model kwarg), so
    every sampling method is compared at the same budget.
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"
    # (5) UQ CONTRACT: engine calls predict() N times (or predict_ensemble once) and aggregates.
    SUPPORTS_UQ: ClassVar[bool] = True
    UQ_METHOD: ClassVar[str] = "sampling"

    def __init__(self) -> None:
        self._models: list[torch.nn.Module] = (
            []
        )  # one for MC-Dropout, K for an ensemble
        self._stats: dict[str, Any] | None = None
        self._device: str = "cpu"

    @property
    def output_location(self) -> OutputLocation:
        """Where predictions live (cell-centered surface, here)."""
        return self.OUTPUT_LOCATION

    def load(
        self, checkpoint_path: str, stats_path: str, device: str, **kwargs: Any
    ) -> "ExampleSamplingUQWrapper":
        """Load the stochastic model(s) used to draw repeated inference passes."""
        self._device = device
        # MC-Dropout: load ONE model and keep its dropout layers in train() mode (stochastic)
        # while the rest is eval(). Ensemble: load one model per kwargs["member_checkpoints"].
        #   model.load_state_dict(torch.load(checkpoint_path, weights_only=True)); model.eval()
        #   for m in model.modules():
        #       if isinstance(m, torch.nn.Dropout): m.train()
        self._stats = load_global_stats(stats_path, device)
        log_inference("example_sampling_uq", f"Loaded from {checkpoint_path}")
        return self

    def prepare_inputs(self, case: CanonicalCase) -> torch.Tensor:
        """Read the case mesh and return the model-ready input tensor."""
        mesh = pv.read(case.mesh_path)
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()
        coords = np.array(mesh.cell_centers().points, dtype=np.float32)
        return torch.tensor(coords, device=self._device)

    def predict(self, model_input: torch.Tensor) -> dict[str, torch.Tensor]:
        """ONE stochastic pass. Must differ call-to-call (dropout mask / weight sample)."""
        with torch.no_grad():
            n = model_input.shape[0]
            # raw = self._models[0](model_input)  # dropout re-samples its mask each call
            raw = {
                "pressure": torch.randn(n, device=self._device),
                "shear_stress": torch.randn((n, 3), device=self._device),
            }
        return raw

    def predict_deterministic(
        self, model_input: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """One DETERMINISTIC pass for ``run.uq.enabled=false`` (no dropout / one member).

        The default ``CFDModel.predict_deterministic`` just calls ``predict``; override it whenever
        ``predict`` is stochastic. MC-Dropout: flip the dropout modules to ``eval()`` for one pass
        then restore them. Ensemble: return a single member. Otherwise turning UQ off would still
        return one random draw instead of a point prediction.
        """
        # MC-Dropout sketch:
        #   dropouts = [m for m in self._models[0].modules() if isinstance(m, torch.nn.Dropout)]
        #   were_training = [m.training for m in dropouts]
        #   for m in dropouts: m.eval()
        #   try: return self.predict(model_input)
        #   finally:
        #       for m, t in zip(dropouts, were_training): m.train(t)
        return self.predict(model_input)

    def predict_ensemble(
        self, model_input: torch.Tensor, n: int
    ) -> Optional[Iterable[dict[str, torch.Tensor]]]:
        """Optional multi-pass path: YIELD each pass/member (prefer a lazy generator).

        Return an ``Iterable[RawOutput]`` (ideally a generator, so only one output is device-
        resident at a time — a ``list`` materializes all ``n`` at once and can OOM). Honor ``n``:
        yield ``n`` stochastic passes for a per-model sampler, or ``min(n, member_count)`` members
        for a fixed-size ensemble (it cannot fabricate more distinct members than it holds). Return
        ``None`` to let the engine fall back to calling ``predict`` ``n`` times.
        """
        if not self._models:
            return None
        k = min(
            n, len(self._models)
        )  # ensemble: honor the budget, capped at member count
        return (self.predict(model_input) for _ in range(k))

    def decode_outputs(
        self,
        raw_output: dict[str, torch.Tensor],
        case: CanonicalCase,
        model_input: Optional[torch.Tensor] = None,
    ) -> dict[str, np.ndarray]:
        """Denormalize ONE pass to physical predictions (the engine aggregates across passes)."""
        p = raw_output["pressure"] * self._std("pressure") + self._mean("pressure")
        w = raw_output["shear_stress"] * self._std("shear_stress") + self._mean(
            "shear_stress"
        )
        return build_predictions_dict(
            pressure=p.cpu().numpy(), shear_stress=w.cpu().numpy().reshape(-1, 3)
        )

    def _mean(self, field: str) -> Any:
        return 0.0 if self._stats is None else self._stats["mean"].get(field, 0.0)

    def _std(self, field: str) -> Any:
        return 1.0 if self._stats is None else self._stats["std"].get(field, 1.0)


# Registration: do this once at import time so the engine can resolve the name.
register_model("example_surface", ExampleSurfaceWrapper)
register_model("example_volume", ExampleVolumeWrapper)
register_model("example_analytic_uq", ExampleAnalyticUQWrapper)
register_model("example_sampling_uq", ExampleSamplingUQWrapper)
