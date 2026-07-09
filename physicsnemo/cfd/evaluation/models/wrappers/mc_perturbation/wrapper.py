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

"""Monte-Carlo **weight-perturbation** proxy for a sampling UQ method (surface).

.. warning::
   This is a **throwaway plumbing proxy**, *not* a valid UQ method. It exists only to exercise
   the ``UQ_METHOD="sampling"`` engine path (N ``predict`` calls, per-pass seeding, streaming
   Welford aggregation, pooled metrics, overlaid reports) *before* a real MC-Dropout wrapper is
   available — **no training and no new checkpoint required** (UQ design doc §8.2). The
   uncertainties it produces are **not calibrated or physically meaningful**. Swap it for the
   real ``mc_dropout_surface`` wrapper (same config) once that lands.

Completely independent :class:`CFDModel` subclass (no inheritance from the deterministic
GeoTransolver wrappers). It composes the shared GeoTransolver + ``TransolverDataPipe`` plumbing in
:mod:`physicsnemo.cfd.evaluation.models.common_wrapper_utils.geotransolver_runtime` for its
deterministic forward, and makes it stochastic in :meth:`predict`: before each pass it multiplies a
subset of weights by ``(1 + eps)`` with ``eps ~ N(0, perturb_std^2)`` (a zero-cost SWAG-flavored
pseudo-ensemble), runs the forward, then restores the weights. The engine drives the N passes and
aggregates them into a
:class:`~physicsnemo.cfd.evaluation.datasets.schema.FieldDistribution` (sample mean +
across-pass std = epistemic).

It decorates ``transformer_models`` (physical-target) checkpoints, so predictions are
re-standardized only — no dynamic-pressure re-dimensionalization
(:attr:`REDIMENSIONALIZE_OUTPUTS` = ``False``).

Model kwargs (``model.kwargs``):

- ``perturb_std`` (float, default ``0.02``): stddev of the multiplicative Gaussian weight noise.
  ``perturb_std → 0`` collapses to the deterministic model (epistemic std → 0), a useful sanity test.
- ``perturb_scope`` (str, default ``"last_n_layers=4"``): which weights to perturb —
  ``"all"``, ``"last_n_layers=K"``, or ``"fraction=f"`` (last fraction ``f`` of parameter tensors).
- plus the shared GeoTransolver runtime kwargs (``cuda_bf16_autocast``; datapipe overrides; etc.);
  ``checkpoint`` / ``stats_path`` are the trained baseline.
"""

import logging
import re
from typing import Any, ClassVar, Literal, Optional

import torch

from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    InferenceDomain,
    coerce_inference_domain_or_default,
)
from physicsnemo.cfd.evaluation.common.io import (
    load_transolver_surface_factors,
    load_transolver_volume_factors,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference
from physicsnemo.cfd.evaluation.models.common_wrapper_utils.geotransolver_runtime import (
    GeoTransolverRuntimeConfig,
    build_geotransolver_backbone,
    build_transolver_batch,
    decode_surface_predictions,
    decode_volume_predictions,
    geotransolver_available,
    geotransolver_forward,
    parse_runtime_kwargs,
    unscale_targets,
)
from physicsnemo.cfd.evaluation.models.model_registry import (
    CFDModel,
    ModelInput,
    OutputLocation,
    Predictions,
    RawOutput,
)

_LOG = logging.getLogger(__name__)


def _parse_perturb_scope(scope: str, named_params: list[str]) -> set[str]:
    """Resolve which parameter names to perturb from a ``perturb_scope`` string.

    Supports ``"all"``, ``"last_n_layers=K"`` (params whose max embedded layer index is within
    the top ``K`` distinct indices), and ``"fraction=f"`` (last fraction of parameter tensors).
    Falls back to the last 25% of tensors when a layer-based scope finds no numeric indices.
    """
    scope = (scope or "").strip()
    if scope in ("", "all"):
        return set(named_params)

    if scope.startswith("fraction="):
        try:
            frac = float(scope.split("=", 1)[1])
        except ValueError:
            frac = 0.25
        frac = min(max(frac, 0.0), 1.0)
        k = max(1, int(round(frac * len(named_params))))
        return set(named_params[-k:])

    if scope.startswith("last_n_layers="):
        try:
            k = int(scope.split("=", 1)[1])
        except ValueError:
            k = 4

        def _layer_index(name: str) -> int | None:
            # Largest integer among dotted segments that are pure digits (the block index).
            ids = [int(t) for t in re.split(r"[.\[\]]", name) if t.isdigit()]
            return max(ids) if ids else None

        indexed = {n: _layer_index(n) for n in named_params}
        distinct = sorted({i for i in indexed.values() if i is not None})
        if distinct:
            keep = set(distinct[-k:])
            return {n for n, i in indexed.items() if i in keep}
        # No numeric layer ids -> fall back to last 25% of tensors.

    k = max(1, int(round(0.25 * len(named_params))))
    return set(named_params[-k:])


class MCPerturbationDrivAerStarWrapper(CFDModel):
    """Weight-perturbation sampling proxy over the baseline GeoTransolver (surface).

    Decorates a ``transformer_models`` (physical-target) checkpoint, so predictions are
    re-standardized only (no dynamic-pressure re-dimensionalization):
    :attr:`REDIMENSIONALIZE_OUTPUTS` = ``False``.
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain | None] = None
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"
    REDIMENSIONALIZE_OUTPUTS: ClassVar[bool] = False
    SUPPORTS_UQ: ClassVar[bool] = True
    UQ_METHOD: ClassVar[str] = "sampling"

    @property
    def output_location(self) -> OutputLocation:
        """See :attr:`CFDModel.output_location` (GeoTransolver predicts at cell centers)."""
        return self.OUTPUT_LOCATION

    @classmethod
    def inference_domain_from_kwargs(
        cls, kwargs: dict[str, Any]
    ) -> InferenceDomain | None:
        """Align benchmark routing with :meth:`load` (default surface when omitted)."""
        return coerce_inference_domain_or_default(
            kwargs.get("inference_domain"),
            default="surface",
            parameter="model.kwargs.inference_domain",
        )

    def __init__(self) -> None:
        self._model: Any = None
        self._datapipe: Any = None
        self._datapipe_geometry_effective: Optional[int] = None
        self._surface_factors: Any = None
        self._volume_factors: Any = None
        self._inference_mode: Literal["surface", "volume"] = "surface"
        self._volume_length_scale: Optional[float] = None
        self._cfg: GeoTransolverRuntimeConfig = GeoTransolverRuntimeConfig()
        self._perturb_std: float = 0.02
        self._perturb_scope: str = "last_n_layers=4"
        self._perturb_names: set[str] = set()

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "MCPerturbationDrivAerStarWrapper":
        """Load the baseline GeoTransolver, then pick the weights to perturb per ``perturb_scope``."""
        if not geotransolver_available():
            raise RuntimeError(
                "MCPerturbationDrivAerStarWrapper requires physicsnemo (GeoTransolver, "
                "TransolverDataPipe, load_checkpoint)."
            )
        kw = dict(kwargs)
        self._perturb_std = float(kw.pop("perturb_std", 0.02))
        self._perturb_scope = str(kw.pop("perturb_scope", "last_n_layers=4"))
        self._inference_mode = coerce_inference_domain_or_default(
            kw.pop("inference_domain", None),
            default="surface",
            parameter="model.kwargs.inference_domain",
        )
        self._cfg = parse_runtime_kwargs(kw, device)

        if self._inference_mode == "volume":
            log_inference(
                "mc_perturbation", f"Loading volume normalization from {stats_path}"
            )
            self._volume_factors = load_transolver_volume_factors(stats_path, device)
            if self._volume_factors is None:
                raise FileNotFoundError(
                    "Volume inference requires ``global_stats.json`` or "
                    f"``volume_fields_normalization.npz`` (looked under {stats_path!r})."
                )
            self._surface_factors = None
        else:
            log_inference(
                "mc_perturbation", f"Loading surface normalization from {stats_path}"
            )
            self._surface_factors = load_transolver_surface_factors(stats_path, device)
            self._volume_factors = None

        self._datapipe = None
        self._datapipe_geometry_effective = None
        self._model = build_geotransolver_backbone(
            checkpoint_path=checkpoint_path,
            device=device,
            inference_mode=self._inference_mode,
        )

        _LOG.warning(
            "MCPerturbationDrivAerStarWrapper is a THROWAWAY sampling PROXY (weight perturbation, "
            "perturb_std=%s, perturb_scope=%r); its uncertainties are NOT calibrated or "
            "physically meaningful. Replace with the real MC-Dropout wrapper for valid UQ.",
            self._perturb_std,
            self._perturb_scope,
        )
        names = [n for n, _ in self._model.named_parameters()]
        self._perturb_names = _parse_perturb_scope(self._perturb_scope, names)
        _LOG.info(
            "Perturbing %d / %d parameter tensors per pass.",
            len(self._perturb_names),
            len(names),
        )
        return self

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        """Build the surface/volume data dict, lazily (re)create the datapipe, and run it."""
        if self._model is None:
            raise RuntimeError("MCPerturbationDrivAerStarWrapper: call load() first")
        result = build_transolver_batch(
            case=case,
            inference_mode=self._inference_mode,
            cfg=self._cfg,
            surface_factors=self._surface_factors,
            volume_factors=self._volume_factors,
            datapipe=self._datapipe,
            geometry_effective=self._datapipe_geometry_effective,
        )
        self._datapipe = result.datapipe
        self._datapipe_geometry_effective = result.geometry_effective
        if result.volume_length_scale is not None:
            self._volume_length_scale = result.volume_length_scale
        return {"batch": result.batch, "datapipe": result.datapipe}

    def _deterministic_predict(self, model_input: ModelInput) -> RawOutput:
        """The unperturbed forward + target unscaling (shared runtime helpers)."""
        raw = geotransolver_forward(
            model=self._model,
            batch=model_input["batch"],
            batch_resolution=self._cfg.batch_resolution,
            cuda_bf16_autocast_enabled=self._cfg.cuda_bf16_autocast,
            device=self._cfg.device,
        )
        return unscale_targets(
            datapipe=model_input["datapipe"],
            predictions=raw,
            batch=model_input["batch"],
            inference_mode=self._inference_mode,
        )

    def predict(self, model_input: ModelInput) -> RawOutput:
        """One stochastic pass: perturb selected weights, run the forward, restore weights.

        Uses the global torch RNG (seeded per pass by the engine's sampling loop) so passes are
        distinct and reproducible. ``perturb_std == 0`` reproduces the deterministic forward.
        """
        if self._model is None or self._datapipe is None:
            raise RuntimeError("MCPerturbationDrivAerStarWrapper: call load() first")

        saved: dict[str, torch.Tensor] = {}
        if self._perturb_std > 0.0 and self._perturb_names:
            with torch.no_grad():
                for name, p in self._model.named_parameters():
                    if name in self._perturb_names:
                        saved[name] = p.detach().clone()
                        eps = torch.randn_like(p) * self._perturb_std
                        p.mul_(1.0 + eps)
        try:
            return self._deterministic_predict(model_input)
        finally:
            if saved:
                with torch.no_grad():
                    for name, p in self._model.named_parameters():
                        if name in saved:
                            p.copy_(saved[name])

    def decode_outputs(
        self,
        raw_output: RawOutput,
        case: CanonicalCase,
        model_input: Optional[ModelInput] = None,
    ) -> Predictions:
        """Emit canonical surface/volume keys in physical units (no ``ρu²`` re-dimensionalization)."""
        if self._inference_mode == "volume":
            return decode_volume_predictions(
                raw_output,
                redimensionalize=self.REDIMENSIONALIZE_OUTPUTS,
                air_density=self._cfg.air_density,
                stream_velocity=self._cfg.stream_velocity,
                length_scale=self._volume_length_scale,
            )
        return decode_surface_predictions(
            raw_output,
            redimensionalize=self.REDIMENSIONALIZE_OUTPUTS,
            air_density=self._cfg.air_density,
            stream_velocity=self._cfg.stream_velocity,
        )
