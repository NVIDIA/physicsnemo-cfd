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

"""Deep-/snapshot-ensemble sampling UQ wrapper for GeoTransolver (surface).

An ensemble aggregates the predictions of several independently-saved model checkpoints (a sibling
sampling UQ method to :mod:`..mc_dropout`, which instead resamples dropout masks on one model). The
engine drives it through the SAME ``UQ_METHOD="sampling"`` path — but here each "pass" is a genuine
model, so the across-member spread
is a meaningful epistemic uncertainty (subject to the caveat below).

.. note::
   As configured for the benchmark, the members are the **last K checkpoints of a single training
   run** (a *snapshot ensemble*), NOT K independently-initialized runs (a *deep ensemble*). Snapshot
   members are correlated, so the ensemble tends to be **under-dispersed** (over-confident) relative
   to a true deep ensemble — but it needs no extra training and is a concrete, honest multi-model
   example. Point ``checkpoint`` at checkpoints from separate runs (via ``member_checkpoints``) for a
   true deep ensemble.

Completely independent :class:`CFDModel` subclass (no inheritance from the other GeoTransolver
wrappers). It composes the shared GeoTransolver + ``TransolverDataPipe`` plumbing in
:mod:`physicsnemo.cfd.evaluation.models.common_wrapper_utils.geotransolver_runtime`, holding one
backbone per member. :meth:`predict_ensemble` runs every member on the shared per-case batch and
**yields** their raw outputs one at a time (a generator, so device memory stays O(field) rather than
O(K × field)); the engine's
:func:`~physicsnemo.cfd.evaluation.benchmarks.uq_inference.run_sampling_inference` folds each into a
streaming Welford mean + across-member (epistemic) std. All members share one per-case block
partition so the spread is model-only. Because ``predict_ensemble`` yields exactly the member count,
``run.uq.num_samples`` is ignored for this row.

It decorates ``transformer_models`` (physical-target) checkpoints, so predictions are
re-standardized only — no dynamic-pressure re-dimensionalization
(:attr:`REDIMENSIONALIZE_OUTPUTS` = ``False``).

Model kwargs (``model.kwargs``):

- ``member_checkpoints`` (list[str], **required**): the explicit member checkpoint files, one per
  ensemble member (any number ``K``). Each must be a specific checkpoint file whose name encodes an
  epoch (``checkpoint.0.<epoch>.pt`` / ``<Model>.0.<epoch>.mdlus``). List members from a single run
  (snapshot ensemble) or from separate runs (true deep ensemble) — same code path either way.
- plus the shared GeoTransolver runtime kwargs (``batch_resolution``, ``geometry_sampling``,
  ``cuda_bf16_autocast``; datapipe overrides; etc.); ``stats_path`` is the shared normalization.

The top-level ``checkpoint`` field is still required by the benchmark (it anchors the run's asset
identity / cache fingerprint); point it at any one of the members (e.g. the final epoch).
"""

import logging
from pathlib import Path
from typing import Any, ClassVar, Iterator, Literal, Optional

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

_LOG = logging.getLogger(__name__)


class GeoTransolverEnsembleDrivAerStarWrapper(CFDModel):
    """K-member (snapshot/deep) ensemble over GeoTransolver checkpoints (surface).

    Decorates ``transformer_models`` (physical-target) checkpoints, so predictions are
    re-standardized only (no dynamic-pressure re-dimensionalization):
    :attr:`REDIMENSIONALIZE_OUTPUTS` = ``False``.
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain | None] = None
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"
    REDIMENSIONALIZE_OUTPUTS: ClassVar[bool] = False
    # UQ contract with the engine: "I produce uncertainty via repeated passes." predict_ensemble()
    # returns one pass per member, and the engine aggregates the spread into mean + epistemic std.
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
        self._models: list[Any] = []
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
    ) -> "GeoTransolverEnsembleDrivAerStarWrapper":
        """Build one GeoTransolver backbone per explicitly-listed member checkpoint."""
        if not geotransolver_available():
            raise RuntimeError(
                "GeoTransolverEnsembleDrivAerStarWrapper requires physicsnemo (GeoTransolver, "
                "TransolverDataPipe, load_checkpoint)."
            )
        kw = dict(kwargs)
        member_checkpoints = kw.pop("member_checkpoints", None)
        self._inference_mode = coerce_inference_domain_or_default(
            kw.pop("inference_domain", None),
            default="surface",
            parameter="model.kwargs.inference_domain",
        )
        self._cfg = parse_runtime_kwargs(kw, device)

        if self._inference_mode == "volume":
            log_inference("ensemble", f"Loading volume normalization from {stats_path}")
            self._volume_factors = load_transolver_volume_factors(stats_path, device)
            if self._volume_factors is None:
                raise FileNotFoundError(
                    "Volume inference requires ``global_stats.json`` or "
                    f"``volume_fields_normalization.npz`` (looked under {stats_path!r})."
                )
            self._surface_factors = None
        else:
            log_inference(
                "ensemble", f"Loading surface normalization from {stats_path}"
            )
            self._surface_factors = load_transolver_surface_factors(stats_path, device)
            self._volume_factors = None

        if not member_checkpoints:
            raise ValueError(
                "GeoTransolverEnsembleDrivAerStarWrapper requires `member_checkpoints`: an explicit list of "
                "checkpoint files (one per member). Auto-discovery from a directory is not "
                "supported — list every member path in model.kwargs.member_checkpoints."
            )
        members = [str(p) for p in member_checkpoints]

        self._datapipe = None
        self._datapipe_geometry_effective = None
        # One backbone per member. Members are ~O(100 MB) each; a handful fit comfortably on device.
        # Each is loaded from its own checkpoint file (strict epoch-in-name resolution).
        self._models = [
            build_geotransolver_backbone(
                checkpoint_path=member,
                device=device,
                inference_mode=self._inference_mode,
            )
            for member in members
        ]
        _LOG.info(
            "GeoTransolverEnsembleDrivAerStarWrapper: loaded %d members from %s",
            len(self._models),
            [Path(m).name for m in members],
        )
        return self

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        """Build the surface/volume batch once per case (shared by every member)."""
        if not self._models:
            raise RuntimeError(
                "GeoTransolverEnsembleDrivAerStarWrapper: call load() first"
            )
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

    def _forward_member(
        self, model: Any, model_input: ModelInput, perm: Any = None
    ) -> RawOutput:
        """Run one member's forward + target unscaling on the shared per-case batch.

        ``perm`` fixes the forward block-partition so every member sees the same point grouping;
        the across-member spread then reflects genuine model differences, not partition noise.
        """
        raw = geotransolver_forward(
            model=model,
            batch=model_input["batch"],
            batch_resolution=self._cfg.batch_resolution,
            cuda_bf16_autocast_enabled=self._cfg.cuda_bf16_autocast,
            device=self._cfg.device,
            perm=perm,
        )
        return unscale_targets(
            datapipe=model_input["datapipe"],
            predictions=raw,
            batch=model_input["batch"],
            inference_mode=self._inference_mode,
        )

    def predict_ensemble(
        self, model_input: ModelInput, n: int
    ) -> Optional[Iterator[RawOutput]]:
        """Yield one raw output per member (lazy generator), sharing one per-case permutation.

        ``n`` (``run.uq.num_samples``) is intentionally ignored: the ensemble has a fixed number of
        members and the engine iterates over exactly what is yielded. This is a **generator**, so
        only one member's output is resident at a time — the engine's Welford accumulator consumes
        each before the next is produced (O(field) device memory, not O(K × field)). All members use
        the same fixed block-partition permutation so the spread is model-only.
        """
        if not self._models or self._datapipe is None:
            raise RuntimeError(
                "GeoTransolverEnsembleDrivAerStarWrapper: call load() first"
            )
        perm = make_forward_permutation(model_input["batch"])
        return (
            self._forward_member(model, model_input, perm) for model in self._models
        )

    def predict(self, model_input: ModelInput) -> RawOutput:
        """Deterministic fallback (first member). The engine uses :meth:`predict_ensemble` for UQ."""
        if not self._models or self._datapipe is None:
            raise RuntimeError(
                "GeoTransolverEnsembleDrivAerStarWrapper: call load() first"
            )
        return self._forward_member(self._models[0], model_input)

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
