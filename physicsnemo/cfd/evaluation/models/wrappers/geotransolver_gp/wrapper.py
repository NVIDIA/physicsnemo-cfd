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

"""GeoTransolver + Gaussian-Process field-head wrapper (analytic surface UQ).

This is the ``UQ_METHOD="analytic"`` archetype and a **completely independent** :class:`CFDModel`
subclass (no inheritance from the deterministic GeoTransolver wrappers). It reuses the shared
GeoTransolver + ``TransolverDataPipe`` plumbing in
:mod:`physicsnemo.cfd.evaluation.models.common_wrapper_utils.geotransolver_runtime` (backbone
construction, VTP → model inputs) and swaps the deterministic readout for a pointwise multitask GP
head (:class:`physicsnemo.experimental.uq.FieldGPHead`): the GP posterior mean is the per-point
surface field and the posterior variance is the per-point uncertainty, in **one forward pass**. It
is a lift-and-shift of the standalone GP inference path
(``examples/.../transformer_models/src/inference_field_gp_zarr._predict_chunked``).

It targets ``transformer_models`` (physical-target) checkpoints, so predictions are re-standardized
only — no dynamic-pressure re-dimensionalization (:attr:`REDIMENSIONALIZE_OUTPUTS` = ``False``).

``decode_distribution`` returns, per canonical field, a
:class:`~physicsnemo.cfd.evaluation.datasets.schema.FieldDistribution` with ``mean``, total
``std`` (epistemic + observation noise), ``epistemic_std``, and ``aleatoric_std`` — all in
**physical units** (both mean and the std channels are denormalized here), so the pooled UQ
metrics consume them directly.

Surface only (the GP head predicts 4 surface tasks: pressure + 3 wall-shear components).

This wrapper loads **two** checkpoints — the GeoTransolver backbone and the ``FieldGPHead`` — and
both are named explicitly:

- ``checkpoint`` must point at a **specific backbone file** whose name encodes the epoch
  (``GeoTransolver.0.<epoch>.mdlus``); the epoch is parsed from that name. Pointing at a directory
  (or an epoch-less name) is rejected — this removes the old ``checkpoint_epoch`` kwarg and the risk
  of the backbone drifting to a "latest" epoch.
- ``gp_head_checkpoint`` should point at the matching ``FieldGPHead.0.<epoch>.pt``. The **exact
  file named here is loaded** (via :func:`physicsnemo.utils.load_model_weights`), so a missing or
  mistyped path raises rather than silently leaving the head randomly initialized (no cross-
  validation against the backbone — passing matching checkpoints is the caller's responsibility).
  It is **optional**: when omitted the wrapper falls back to the backbone's sibling
  ``FieldGPHead.0.<epoch>.pt`` (which must exist). Either way the exact head file it loads is logged.

Model kwargs (``model.kwargs`` in config), matching the trained checkpoint's GP settings:

- ``gp_head_checkpoint`` (path): the ``FieldGPHead.0.<epoch>.pt`` to load (see above).
- ``gp_feature_norm`` (``"none"`` | ``"l2"`` | ``"layernorm"`` | ``"l2_radial"``): must match training.
- ``gp_lengthscale_range`` / ``gp_lengthscale_prior`` / ``gp_outputscale_prior``: GP kernel config.
- ``gp_n_inducing`` (default 256), ``gp_mlp_hidden`` (optional DKL MLP), ``num_tasks`` (default 4).
- ``gp_spectral_norm_coeff`` (float, default 0.0) / ``gp_dkl_residual`` (bool, default True):
  DUE-style bi-Lipschitz DKL. When ``gp_spectral_norm_coeff > 0`` the head uses a spectral-
  normalised + residual feature extractor (van Amersfoort et al., 2021) instead of the plain DKL
  MLP; both must match training. The defaults reproduce the plain MLP (non-DUE) path unchanged.
- ``gp_inference_chunk_size`` (default 51200): points per backbone+GP chunk (drawn from a random
  permutation, then inverted — see the standalone script for why contiguous chunks degrade preds).
- ``cuda_bf16_autocast`` (bool, default False): run forwards under bf16 autocast.
"""

from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import torch

from physicsnemo.cfd.evaluation.common.checkpoint_compat import (
    trusted_torch_load_context,
)
from physicsnemo.cfd.evaluation.common.io import load_transolver_surface_factors
from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    FieldDistribution,
    InferenceDomain,
    build_predictions_dict,
    build_predictive_distribution,
    coerce_inference_domain_or_default,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference
from physicsnemo.cfd.evaluation.models.common_wrapper_utils.geotransolver_runtime import (
    GeoTransolverRuntimeConfig,
    build_geotransolver_backbone,
    build_transolver_batch,
    geotransolver_available,
    global_fx_to_bnc,
    parse_runtime_kwargs,
    resolve_checkpoint_file,
)
from physicsnemo.cfd.evaluation.models.inference_autocast import cuda_bf16_autocast
from physicsnemo.cfd.evaluation.models.model_registry import (
    CFDModel,
    ModelInput,
    OutputLocation,
    Predictions,
    RawOutput,
)

try:
    from physicsnemo.experimental.uq import FieldGPHead
    from physicsnemo.utils import load_model_weights

    _GP_AVAILABLE = True
except ImportError:
    _GP_AVAILABLE = False

#: Surface field channels predicted by the GP head: pressure, wall-shear (x, y, z).
NUM_SURFACE_TASKS = 4


class GeoTransolverGPDrivAerStarWrapper(CFDModel):
    """GeoTransolver backbone + ``FieldGPHead`` for analytic per-point surface UQ.

    Trained on ``transformer_models`` (physical-target) checkpoints, so predictions are
    re-standardized only (no dynamic-pressure re-dimensionalization):
    :attr:`REDIMENSIONALIZE_OUTPUTS` = ``False``. Completely independent of the deterministic
    GeoTransolver wrappers (subclasses ``CFDModel`` directly, composing the shared runtime helpers).
    """

    REDIMENSIONALIZE_OUTPUTS: ClassVar[bool] = False
    INFERENCE_DOMAIN: ClassVar[InferenceDomain | None] = "surface"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"
    SUPPORTS_UQ: ClassVar[bool] = True
    UQ_METHOD: ClassVar[str] = "analytic"

    @property
    def output_location(self) -> OutputLocation:
        """See :attr:`CFDModel.output_location` (GeoTransolver predicts at cell centers)."""
        return self.OUTPUT_LOCATION

    @classmethod
    def inference_domain_from_kwargs(
        cls, kwargs: dict[str, Any]
    ) -> InferenceDomain | None:
        """Surface only; reject volume explicitly (the GP head is a 4-task surface head)."""
        dom = coerce_inference_domain_or_default(
            kwargs.get("inference_domain"),
            default="surface",
            parameter="model.kwargs.inference_domain",
        )
        if dom != "surface":
            raise NotImplementedError(
                "GeoTransolverGPDrivAerStarWrapper supports surface inference only; got "
                f"inference_domain={dom!r}."
            )
        return dom

    def __init__(self) -> None:
        self._model: Any = None
        self._datapipe: Any = None
        self._datapipe_geometry_effective: Optional[int] = None
        self._surface_factors: Any = None
        self._cfg: GeoTransolverRuntimeConfig = GeoTransolverRuntimeConfig()
        self._head: Optional[FieldGPHead] = None
        self._gp_kw: dict[str, Any] = {}
        self._checkpoint_epoch: Optional[int] = None
        self._head_checkpoint: Optional[str] = None
        self._gp_chunk_size: int = 51200
        self._field_mean_t: Optional[torch.Tensor] = None
        self._field_std_t: Optional[torch.Tensor] = None

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "GeoTransolverGPDrivAerStarWrapper":
        """Build the GeoTransolver backbone + surface factors, and prepare the GP head.

        The backbone is constructed via the shared runtime helpers; the ``FieldGPHead`` is built
        lazily on the first :meth:`predict` (its input dim is probed from the backbone features).
        """
        if not _GP_AVAILABLE or not geotransolver_available():
            raise RuntimeError(
                "GeoTransolverGPDrivAerStarWrapper requires physicsnemo with GeoTransolver + "
                "physicsnemo.experimental.uq.FieldGPHead."
            )
        kw = dict(kwargs)
        # Surface-only; validate and force the routing hint before building the datapipe.
        self.inference_domain_from_kwargs(kw)
        kw.pop("inference_domain", None)

        # GP-head hyperparameters (must match the trained checkpoint). Pulled off here so they
        # are not consumed by the shared runtime-kwargs parsing.
        self._gp_chunk_size = int(kw.pop("gp_inference_chunk_size", 51200))
        # Explicit GP-head checkpoint file (optional). When omitted we fall back to the backbone's
        # sibling ``FieldGPHead.0.<epoch>.pt``. No cross-checking against the backbone — passing a
        # matching pair is the caller's responsibility.
        head_ckpt_arg = kw.pop("gp_head_checkpoint", None)
        self._gp_kw = {
            "num_tasks": int(kw.pop("num_tasks", NUM_SURFACE_TASKS)),
            "n_inducing": int(kw.pop("gp_n_inducing", 256)),
            "mlp_hidden": (
                list(kw.pop("gp_mlp_hidden"))
                if kw.get("gp_mlp_hidden") is not None
                else kw.pop("gp_mlp_hidden", None)
            ),
            "lengthscale_range": tuple(kw.pop("gp_lengthscale_range", (0.01, 10.0))),
            "lengthscale_prior": (
                tuple(kw.pop("gp_lengthscale_prior"))
                if kw.get("gp_lengthscale_prior") is not None
                else kw.pop("gp_lengthscale_prior", None)
            ),
            "outputscale_prior": (
                tuple(kw.pop("gp_outputscale_prior"))
                if kw.get("gp_outputscale_prior") is not None
                else kw.pop("gp_outputscale_prior", None)
            ),
            "feature_norm": str(kw.pop("gp_feature_norm", "none")),
            # DUE-style bi-Lipschitz DKL extractor (van Amersfoort et al., 2021). When
            # ``gp_spectral_norm_coeff > 0`` the head builds a spectral-normalised + residual
            # feature extractor instead of the plain DKL MLP, so this MUST match training or the
            # ``FieldGPHead`` state dict will not reconstruct. Defaults (0.0 / True) keep the plain
            # MLP path, so existing (non-DUE) GP checkpoints are unaffected.
            "spectral_norm_coeff": float(kw.pop("gp_spectral_norm_coeff", 0.0)),
            "dkl_residual": bool(kw.pop("gp_dkl_residual", True)),
        }

        self._cfg = parse_runtime_kwargs(kw, device)
        log_inference(
            "geotransolver_gp", f"Loading surface normalization from {stats_path}"
        )
        self._surface_factors = load_transolver_surface_factors(stats_path, device)
        if self._surface_factors is None:
            raise FileNotFoundError(
                "GeoTransolverGPDrivAerStarWrapper requires surface normalization "
                "(``global_stats.json`` or ``surface_fields_normalization.npz``) at "
                f"{stats_path!r}."
            )
        # Cache flat (4,) standardization vectors for physical de-normalization of the GP outputs.
        self._field_mean_t = self._surface_factors["mean"].reshape(-1).to(device)
        self._field_std_t = self._surface_factors["std"].reshape(-1).to(device)

        self._datapipe = None
        self._datapipe_geometry_effective = None
        self._model = build_geotransolver_backbone(
            checkpoint_path=checkpoint_path,
            device=device,
            inference_mode="surface",
        )

        # The head is loaded from the EXACT ``gp_head_checkpoint`` file when given (so the file the
        # user names is the one read), else from the backbone's sibling ``FieldGPHead.0.<epoch>.pt``.
        # Either way the file must exist (no cross-validation of contents — passing matching
        # checkpoints is the caller's responsibility).
        if head_ckpt_arg is not None:
            # resolve_checkpoint_file validates existence + the epoch-in-name shape.
            _, self._checkpoint_epoch = resolve_checkpoint_file(str(head_ckpt_arg))
            self._head_checkpoint = str(Path(head_ckpt_arg))
        else:
            ckpt_dir, self._checkpoint_epoch = resolve_checkpoint_file(checkpoint_path)
            self._head_checkpoint = str(
                Path(ckpt_dir) / f"FieldGPHead.0.{self._checkpoint_epoch}.pt"
            )
            if not Path(self._head_checkpoint).is_file():
                raise FileNotFoundError(
                    "GP head checkpoint not found next to the backbone: "
                    f"{self._head_checkpoint!r}. Pass model.kwargs.gp_head_checkpoint explicitly."
                )
        log_inference(
            "geotransolver_gp",
            f"GP checkpoints -> backbone: {checkpoint_path} | head: {self._head_checkpoint}",
        )
        return self

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        """Build the surface data dict, lazily (re)create the datapipe, and run it (surface only)."""
        if self._model is None:
            raise RuntimeError("GeoTransolverGPDrivAerStarWrapper: call load() first")
        result = build_transolver_batch(
            case=case,
            inference_mode="surface",
            cfg=self._cfg,
            surface_factors=self._surface_factors,
            volume_factors=None,
            datapipe=self._datapipe,
            geometry_effective=self._datapipe_geometry_effective,
        )
        self._datapipe = result.datapipe
        self._datapipe_geometry_effective = result.geometry_effective
        return {"batch": result.batch, "datapipe": result.datapipe}

    def _build_and_load_head(self, batch: dict) -> None:
        """Probe the backbone feature dim on ``batch``, build the GP head, load ONLY its weights.

        The backbone was already loaded from its own ``checkpoint`` in :meth:`load`; here we load
        just the ``FieldGPHead`` from the EXACT ``self._head_checkpoint`` file (the one named by
        ``gp_head_checkpoint``, or the validated backbone sibling). Loading only the head from its
        own file avoids reloading — or silently overwriting — the backbone, and a missing file
        raises rather than leaving the head randomly initialized.
        """
        dev = torch.device(self._cfg.device)
        feature_dim = self._probe_feature_dim(batch)
        self._head = FieldGPHead(
            input_dim=feature_dim,
            n_train=1,  # unused at inference
            **self._gp_kw,
        ).to(dev)
        with trusted_torch_load_context():
            load_model_weights(self._head, self._head_checkpoint, device=dev)
        self._model.eval()
        self._head.eval()
        self._head.likelihood.eval()
        log_inference(
            "geotransolver_gp",
            f"Loaded FieldGPHead from {self._head_checkpoint} "
            f"(epoch {self._checkpoint_epoch}); feature_dim={feature_dim}, "
            f"gp_dim={getattr(self._head, 'gp_input_dim', '?')}",
        )

    def _autocast_ctx(self):
        return (
            cuda_bf16_autocast(self._cfg.device)
            if self._cfg.cuda_bf16_autocast
            else nullcontext()
        )

    @torch.no_grad()
    def _probe_feature_dim(self, batch: dict) -> int:
        n = min(batch["embeddings"].shape[1], 1024)
        fx = global_fx_to_bnc(batch["fx"])
        local_emb = batch["embeddings"][:, :n]
        geometry = batch.get("geometry")
        with self._autocast_ctx():
            _, point_features = self._model(
                global_embedding=fx,
                local_embedding=local_emb,
                geometry=geometry,
                local_positions=local_emb[:, :, :3],
                return_point_features=True,
            )
        return int(point_features.shape[-1])

    @torch.no_grad()
    def _predict_chunked_gp(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backbone + GP over all points, chunked on a random permutation (order restored).

        Returns ``(mean, total_std, epistemic_std)`` each ``(N, T)`` on device, in the
        normalized target space the GP was trained in.
        """
        fx = global_fx_to_bnc(batch["fx"])
        geometry = batch.get("geometry")
        embeddings = batch["embeddings"]
        n = embeddings.shape[1]
        chunk = self._gp_chunk_size
        if chunk is None or chunk <= 0 or chunk > n:
            chunk = n

        perm = torch.randperm(n, device=embeddings.device)
        mean_blocks: list[torch.Tensor] = []
        std_blocks: list[torch.Tensor] = []
        epi_blocks: list[torch.Tensor] = []
        with self._autocast_ctx():
            for idx_block in torch.split(perm, chunk):
                local_emb = embeddings[:, idx_block]
                _, point_features = self._model(
                    global_embedding=fx,
                    local_embedding=local_emb,
                    geometry=geometry,
                    local_positions=local_emb[:, :, :3],
                    return_point_features=True,
                )
                pred = self._head.predict(point_features)
                mean_blocks.append(pred.mean)
                std_blocks.append(pred.variance.clamp_min(0).sqrt())
                epi_blocks.append(pred.epistemic_variance.clamp_min(0).sqrt())

        mean_perm = torch.cat(mean_blocks, dim=1)
        std_perm = torch.cat(std_blocks, dim=1)
        epi_perm = torch.cat(epi_blocks, dim=1)
        inverse = torch.empty(n, dtype=torch.long, device=mean_perm.device)
        inverse[perm] = torch.arange(n, device=mean_perm.device)
        mean = mean_perm[:, inverse].squeeze(0)
        std = std_perm[:, inverse].squeeze(0)
        epi = epi_perm[:, inverse].squeeze(0)
        return mean, std, epi

    def predict(self, model_input: ModelInput) -> RawOutput:
        """One forward pass: backbone features → GP posterior (mean, total std, epistemic std)."""
        if self._model is None:
            raise RuntimeError("GeoTransolverGPDrivAerStarWrapper: call load() first")
        batch = model_input["batch"]
        if self._head is None:
            self._build_and_load_head(batch)
        log_inference("geotransolver_gp", "Running GP forward pass (mean + variance)…")
        mean, std, epi = self._predict_chunked_gp(batch)
        return {"mean": mean, "total_std": std, "epistemic_std": epi}

    def _to_physical(
        self, mean: torch.Tensor, std: torch.Tensor, epi: torch.Tensor
    ) -> dict[str, np.ndarray]:
        """De-normalize (mean, total std, epistemic std) from GP target space to physical units.

        For the affine standardization ``phys = (norm * field_std + field_mean) * q`` (with
        ``q`` the dynamic pressure ``rho * u^2``), the mean maps fully while std/variance scale
        by ``|field_std * q|`` only (the additive ``field_mean`` and the affine offset drop out).
        ``q`` collapses to 1 when the checkpoint's targets are already dimensional (the
        ``transformer_models`` GP checkpoints; see ``REDIMENSIONALIZE_OUTPUTS``), so physical
        fields are just ``norm*std + mean`` — matching ``field_gp_utils.compute_drag_uq_stats``.
        """
        q = (
            self._cfg.air_density * (self._cfg.stream_velocity**2)
            if self.REDIMENSIONALIZE_OUTPUTS
            else 1.0
        )
        fm = self._field_mean_t
        fs = self._field_std_t
        scale = fs * q  # (T,)
        mean_phys = (mean * fs + fm) * q  # (N, T)
        std_phys = std * scale
        epi_phys = epi * scale
        # Aleatoric (observation-noise) std from total vs epistemic variance.
        ale_phys = (std_phys**2 - epi_phys**2).clamp_min(0).sqrt()
        return {
            "mean": mean_phys.detach().cpu().numpy().astype(np.float32),
            "std": std_phys.detach().cpu().numpy().astype(np.float32),
            "epistemic_std": epi_phys.detach().cpu().numpy().astype(np.float32),
            "aleatoric_std": ale_phys.detach().cpu().numpy().astype(np.float32),
        }

    def decode_distribution(
        self,
        raw_output: RawOutput,
        case: CanonicalCase,
        model_input: Optional[ModelInput] = None,
    ) -> dict[str, FieldDistribution]:
        """Map GP posterior to per-field :class:`FieldDistribution`s in physical units."""
        phys = self._to_physical(
            raw_output["mean"], raw_output["total_std"], raw_output["epistemic_std"]
        )
        log_inference(
            "geotransolver_gp",
            "Decoding GP distribution (pressure + WSS, physical units)…",
        )
        out: dict[str, FieldDistribution] = {}
        out["pressure"] = build_predictive_distribution(
            mean=phys["mean"][:, 0],
            std=phys["std"][:, 0],
            epistemic_std=phys["epistemic_std"][:, 0],
            aleatoric_std=phys["aleatoric_std"][:, 0],
        )
        out["shear_stress"] = build_predictive_distribution(
            mean=phys["mean"][:, 1:4],
            std=phys["std"][:, 1:4],
            epistemic_std=phys["epistemic_std"][:, 1:4],
            aleatoric_std=phys["aleatoric_std"][:, 1:4],
        )
        return out

    def decode_outputs(
        self,
        raw_output: RawOutput,
        case: CanonicalCase,
        model_input: Optional[ModelInput] = None,
    ) -> Predictions:
        """Point-estimate (GP mean) predictions in physical units (for non-UQ metric paths)."""
        phys = self._to_physical(
            raw_output["mean"], raw_output["total_std"], raw_output["epistemic_std"]
        )
        return build_predictions_dict(
            pressure=phys["mean"][:, 0], shear_stress=phys["mean"][:, 1:4]
        )
