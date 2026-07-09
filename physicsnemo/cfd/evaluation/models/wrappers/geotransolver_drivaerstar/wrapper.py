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

"""GeoTransolver wrapper for the DrivAerStar (``transformer_models``) convention.

This module is **purely additive**: it does not modify the original
:class:`~physicsnemo.cfd.evaluation.models.wrappers.geotransolver.wrapper.GeoTransolverWrapper`
(the DrivAerML / DoMINO non-dimensional path). It only adds a sibling wrapper for checkpoints
trained with the ``examples/cfd/external_aerodynamics/transformer_models`` pipeline, whose
targets are standardized **physical** fields.

Both wrappers are independent :class:`CFDModel` subclasses and compose the shared plumbing in
:mod:`physicsnemo.cfd.evaluation.models.common_wrapper_utils.geotransolver_runtime`.
"""

from typing import Any, ClassVar, Literal, Optional

from physicsnemo.cfd.evaluation.common.io import (
    load_transolver_surface_factors,
    load_transolver_volume_factors,
)
from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    InferenceDomain,
    coerce_inference_domain_or_default,
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


class GeoTransolverDrivAerStarWrapper(CFDModel):
    """GeoTransolver on **DrivAerStar** (``examples/.../transformer_models`` convention, physical targets).

    Registered as ``geotransolver_drivaerstar_surface``. Identical mechanics to
    :class:`~physicsnemo.cfd.evaluation.models.wrappers.geotransolver.wrapper.GeoTransolverWrapper`,
    but the training targets are **standardized physical fields** (``preprocess.py``:
    ``(x-mean)/std``), so ``TransolverDataPipe.unscale_model_targets`` already returns physical units
    (``norm*std + mean``) and :meth:`decode_outputs` must **not** re-dimensionalize by ``ρu²`` —
    matching ``field_gp_utils`` and the DrivAerStar ``.vtk`` ground truth. Completely independent of
    ``GeoTransolverWrapper`` (both subclass ``CFDModel`` and compose the same runtime helpers); only
    :attr:`REDIMENSIONALIZE_OUTPUTS` differs.
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain | None] = None
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"
    REDIMENSIONALIZE_OUTPUTS: ClassVar[bool] = False

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

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "GeoTransolverDrivAerStarWrapper":
        """Load the GeoTransolver backbone + physical-target normalization factors onto ``device``."""
        if not geotransolver_available():
            raise RuntimeError(
                "GeoTransolver wrapper requires physicsnemo (GeoTransolver, "
                "TransolverDataPipe, load_checkpoint)."
            )
        kw = dict(kwargs)
        self._inference_mode = coerce_inference_domain_or_default(
            kw.pop("inference_domain", None),
            default="surface",
            parameter="model.kwargs.inference_domain",
        )
        self._cfg = parse_runtime_kwargs(kw, device)

        if self._inference_mode == "volume":
            log_inference(
                "geotransolver", f"Loading volume normalization from {stats_path}"
            )
            self._volume_factors = load_transolver_volume_factors(stats_path, device)
            if self._volume_factors is None:
                raise FileNotFoundError(
                    "Volume inference requires ``global_stats.json`` (with velocity, "
                    "pressure, turbulent_viscosity) or ``volume_fields_normalization.npz`` "
                    f"next to stats/checkpoint (looked under {stats_path!r})."
                )
            self._surface_factors = None
        else:
            log_inference(
                "geotransolver", f"Loading surface normalization from {stats_path}"
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
        return self

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        """Build the surface/volume data dict, lazily (re)create the datapipe, and run it."""
        if self._model is None:
            raise RuntimeError("GeoTransolverDrivAerStarWrapper: call load() first")
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

    def predict(self, model_input: ModelInput) -> RawOutput:
        """Run blocked GeoTransolver forward passes and return unscaled (physical) targets."""
        if self._model is None or self._datapipe is None:
            raise RuntimeError("GeoTransolverDrivAerStarWrapper: call load() first")
        log_inference("geotransolver", "Running forward pass (predicting fields)…")
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
