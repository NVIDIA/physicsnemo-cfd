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

"""**MC-Dropout** sampling UQ over a Concrete-Dropout GeoTransolver (surface/volume).

This is the MC-Dropout sampling baseline (``geotransolver_mc_dropout_surface``). The backbone was trained with
learned per-layer :class:`~physicsnemo.nn.ConcreteDropout` (``model.concrete_dropout=true``,
Gal-Hron-Kendall 2017), so its dropout rates are calibrated by the training objective rather than
hand-picked. UQ is obtained at inference exactly as in the training repo's ``setup_mc_dropout`` /
``mc_dropout_inference_loop``: put the network in ``eval()`` but keep the Concrete-Dropout layers
**stochastic** (``.train()``), then average over ``N`` forward passes. The engine drives the ``N``
passes (``run.uq.num_samples``) and aggregates the across-pass spread into a
:class:`~physicsnemo.cfd.evaluation.datasets.schema.FieldDistribution` (mean + epistemic std) via
streaming Welford — same ``UQ_METHOD="sampling"`` path as the ensemble wrapper.

Completely independent :class:`CFDModel` subclass (no inheritance from the deterministic
GeoTransolver wrappers). It composes the shared GeoTransolver + ``TransolverDataPipe`` plumbing in
:mod:`physicsnemo.cfd.evaluation.models.common_wrapper_utils.geotransolver_runtime`; the *only*
difference from a deterministic forward is that the Concrete-Dropout submodules stay stochastic, so
:meth:`predict` returns a different draw each call.

It decorates ``transformer_models`` (physical-target) checkpoints, so predictions are
re-standardized only — no dynamic-pressure re-dimensionalization
(:attr:`REDIMENSIONALIZE_OUTPUTS` = ``False``).

Model kwargs (``model.kwargs``):

- the shared GeoTransolver runtime kwargs (``batch_resolution``, ``geometry_sampling``,
  ``cuda_bf16_autocast``; datapipe overrides; etc.). ``checkpoint`` must point at a
  Concrete-Dropout checkpoint file (trained with ``concrete_dropout=true``); ``stats_path`` is the
  shared surface/volume normalization.

The number of stochastic passes is **not** a model kwarg — it is the benchmark-wide
``run.uq.num_samples`` so every sampling method (this and the ensemble) is compared at the same
budget.
"""

import logging
from typing import Any, ClassVar, Literal, Optional

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
    make_forward_permutation,
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

# ConcreteDropout is only needed to (a) find the dropout submodules and (b) read their learned
# rates for a sanity log. Guard the import like geotransolver_runtime guards physicsnemo.
try:
    from physicsnemo.nn import ConcreteDropout, get_concrete_dropout_rates

    _CONCRETE_DROPOUT_AVAILABLE = True
except ImportError:
    _CONCRETE_DROPOUT_AVAILABLE = False

_LOG = logging.getLogger(__name__)


class GeoTransolverMCDropoutDrivAerStarWrapper(CFDModel):
    """MC-Dropout sampling over a Concrete-Dropout GeoTransolver (surface/volume).

    Decorates a ``transformer_models`` (physical-target) checkpoint, so predictions are
    re-standardized only (no dynamic-pressure re-dimensionalization):
    :attr:`REDIMENSIONALIZE_OUTPUTS` = ``False``.
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain | None] = None
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"
    REDIMENSIONALIZE_OUTPUTS: ClassVar[bool] = False
    # Contract with the engine: "I produce uncertainty, get it by calling predict() many times."
    # The engine loops predict() run.uq.num_samples times per case and turns the spread of the
    # stochastic passes into a mean + epistemic std (streaming Welford).
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

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "GeoTransolverMCDropoutDrivAerStarWrapper":
        """Load the Concrete-Dropout backbone and keep its dropout layers stochastic for MC passes."""
        if not geotransolver_available():
            raise RuntimeError(
                "GeoTransolverMCDropoutDrivAerStarWrapper requires physicsnemo (GeoTransolver, "
                "TransolverDataPipe, load_checkpoint)."
            )
        if not _CONCRETE_DROPOUT_AVAILABLE:
            raise RuntimeError(
                "GeoTransolverMCDropoutDrivAerStarWrapper requires physicsnemo.nn.ConcreteDropout "
                "(train the backbone with model.concrete_dropout=true)."
            )
        kw = dict(kwargs)
        self._inference_mode = coerce_inference_domain_or_default(
            kw.pop("inference_domain", None),
            default="surface",
            parameter="model.kwargs.inference_domain",
        )
        self._cfg = parse_runtime_kwargs(kw, device)

        if self._inference_mode == "volume":
            log_inference("mc_dropout", f"Loading volume normalization from {stats_path}")
            self._volume_factors = load_transolver_volume_factors(stats_path, device)
            if self._volume_factors is None:
                raise FileNotFoundError(
                    "Volume inference requires ``global_stats.json`` or "
                    f"``volume_fields_normalization.npz`` (looked under {stats_path!r})."
                )
            self._surface_factors = None
        else:
            log_inference("mc_dropout", f"Loading surface normalization from {stats_path}")
            self._surface_factors = load_transolver_surface_factors(stats_path, device)
            self._volume_factors = None

        self._datapipe = None
        self._datapipe_geometry_effective = None
        # Build WITH concrete_dropout=True so the checkpoint's learned ConcreteDropout parameters
        # (p_logit per layer) load cleanly; a plain backbone would reject those extra keys.
        self._model = build_geotransolver_backbone(
            checkpoint_path=checkpoint_path,
            device=device,
            inference_mode=self._inference_mode,
            concrete_dropout=True,
        )
        self._enable_mc_dropout()
        return self

    def _enable_mc_dropout(self) -> None:
        """Keep the network in eval() but flip ConcreteDropout layers to train() (stochastic).

        This is the MC-Dropout trick: everything deterministic (norms, etc.) stays in eval, but the
        Concrete-Dropout masks are re-sampled every forward pass, so N passes give N draws. Mirrors
        the training repo's ``setup_mc_dropout``.
        """
        self._model.eval()
        n_dropout = 0
        for module in self._model.modules():
            if isinstance(module, ConcreteDropout):
                module.train()
                n_dropout += 1
        if n_dropout == 0:
            raise RuntimeError(
                "No ConcreteDropout layers found in the loaded checkpoint. Was it trained with "
                "model.concrete_dropout=true? MC-Dropout has no source of stochasticity otherwise."
            )
        rates = get_concrete_dropout_rates(self._model)
        if rates:
            vals = list(rates.values())
            _LOG.info(
                "MC-Dropout enabled over %d ConcreteDropout layers; learned rates "
                "min=%.4f max=%.4f mean=%.4f",
                n_dropout,
                min(vals),
                max(vals),
                sum(vals) / len(vals),
            )
        else:
            _LOG.info("MC-Dropout enabled over %d ConcreteDropout layers.", n_dropout)

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        """Build the surface/volume data dict, lazily (re)create the datapipe, and run it."""
        if self._model is None:
            raise RuntimeError("GeoTransolverMCDropoutDrivAerStarWrapper: call load() first")
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
        # Fix ONE forward block-partition per case (reused across all N passes below) so the
        # across-pass spread reflects only the resampled dropout masks, not random point grouping.
        perm = make_forward_permutation(result.batch)
        return {"batch": result.batch, "datapipe": result.datapipe, "perm": perm}

    def predict(self, model_input: ModelInput) -> RawOutput:
        """One stochastic MC-Dropout pass (Concrete-Dropout layers resample their masks).

        No weights are modified: the stochasticity comes entirely from the dropout masks, which are
        redrawn from the global torch RNG (re-seeded per pass by the engine's sampling loop, so
        passes differ from each other yet are reproducible run-to-run). The forward block-partition
        is held fixed across passes (from :meth:`prepare_inputs`) so only the dropout varies.
        """
        if self._model is None or self._datapipe is None:
            raise RuntimeError("GeoTransolverMCDropoutDrivAerStarWrapper: call load() first")
        raw = geotransolver_forward(
            model=self._model,
            batch=model_input["batch"],
            batch_resolution=self._cfg.batch_resolution,
            cuda_bf16_autocast_enabled=self._cfg.cuda_bf16_autocast,
            device=self._cfg.device,
            perm=model_input.get("perm"),
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
