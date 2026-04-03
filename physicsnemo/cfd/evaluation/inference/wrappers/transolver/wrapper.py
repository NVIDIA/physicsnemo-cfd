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

"""Transolver model wrapper for surface or volume inference (TransolverDataPipe + VTK).

Surface: cell-centered pressure + WSS (default). Volume: velocity + pressure + nut on VTU when
``inference_domain: volume``, matching ``inference_on_vtk.py`` / ``transolver_volume`` training.
"""

from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

import numpy as np
import torch

from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    InferenceDomain,
    build_predictions_dict,
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
from physicsnemo.cfd.evaluation.common.checkpoint_compat import trusted_torch_load_context
from physicsnemo.cfd.evaluation.common.io import load_transolver_surface_factors, load_transolver_volume_factors
from physicsnemo.cfd.evaluation.inference.common_wrapper_utils.vtk_datapipe_io import (
    build_surface_data_dict,
    build_volume_data_dict,
    run_id_from_case_id,
)

# Optional physicsnemo imports (required for real inference)
try:
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.datapipes.cae.transolver_datapipe import TransolverDataPipe
    from physicsnemo.models.transolver.transolver import Transolver
    from physicsnemo.utils import load_checkpoint

    _PHYSICSNEMO_AVAILABLE = True
except ImportError:
    _PHYSICSNEMO_AVAILABLE = False


# Default Transolver surface config (from transolver_surface / model/transolver.yaml)
DEFAULT_TRANSOLVER_KW = dict(
    functional_dim=2,
    out_dim=4,
    embedding_dim=6,
    n_layers=20,
    n_hidden=256,
    dropout=0.0,
    n_head=8,
    act="gelu",
    mlp_ratio=2,
    slice_num=512,
    unified_pos=False,
    ref=8,
    structured_shape=None,
    use_te=False,
    time_input=False,
    plus=False,
)

DEFAULT_TRANSOLVER_VOLUME_KW = {
    **DEFAULT_TRANSOLVER_KW,
    "embedding_dim": 7,
    "out_dim": 5,
}

_DATAPIPE_KEYS = frozenset(
    {
        "include_normals",
        "include_sdf",
        "translational_invariance",
        "scale_invariance",
        "reference_scale",
        "broadcast_global_features",
        "include_geometry",
        "return_mesh_features",
    }
)


def _surface_datapipe_kw() -> dict[str, Any]:
    return dict(
        include_normals=True,
        include_sdf=False,
        broadcast_global_features=True,
        include_geometry=False,
        translational_invariance=True,
        reference_scale=[12.0, 4.5, 3.25],
        scale_invariance=True,
        return_mesh_features=True,
    )


def _volume_datapipe_kw() -> dict[str, Any]:
    return dict(
        include_normals=True,
        include_sdf=True,
        translational_invariance=True,
        scale_invariance=True,
        reference_scale=[12.0, 4.5, 3.25],
        broadcast_global_features=True,
        include_geometry=False,
        return_mesh_features=False,
    )


class TransolverWrapper(CFDModel):
    """Transolver: boundary VTP (surface) or volume VTU; set ``model.inference_domain: volume`` for VTU."""

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"

    @property
    def output_location(self) -> OutputLocation:
        return self.OUTPUT_LOCATION

    def __init__(self) -> None:
        self._model: Optional[Transolver] = None
        self._datapipe: Optional[TransolverDataPipe] = None
        self._device: str = "cuda:0"
        self._air_density: float = 1.205
        self._stream_velocity: float = 30.0
        self._batch_resolution: int = 2048
        self._datapipe_resolution: int = 10_000_000
        self._inference_mode: Literal["surface", "volume"] = "surface"
        self._datapipe_user_kw: dict[str, Any] = {}

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "TransolverWrapper":
        if not _PHYSICSNEMO_AVAILABLE:
            raise RuntimeError(
                "Transolver wrapper requires physicsnemo (Transolver, TransolverDataPipe, load_checkpoint)."
            )
        kw = dict(kwargs)
        dom = kw.pop("inference_domain", None)
        self._inference_mode = dom if dom in ("surface", "volume") else "surface"

        self._device = device
        self._air_density = float(kw.get("air_density", 1.205))
        self._stream_velocity = float(kw.get("stream_velocity", 30.0))
        self._batch_resolution = int(kw.get("batch_resolution", 2048))
        self._datapipe_resolution = int(kw.get("resolution", 10_000_000))

        checkpoint_dir = Path(checkpoint_path)
        if checkpoint_dir.is_file():
            checkpoint_dir = checkpoint_dir.parent

        dp_user = {k: kw.pop(k) for k in list(kw.keys()) if k in _DATAPIPE_KEYS}
        self._datapipe_user_kw = dp_user

        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        dev = torch.device(device)

        if self._inference_mode == "volume":
            log_inference("transolver", f"Loading volume normalization from {stats_path}")
            volume_factors = load_transolver_volume_factors(stats_path, device)
            if volume_factors is None:
                raise FileNotFoundError(
                    "Volume inference requires ``global_stats.json`` (with velocity, "
                    "pressure_volume, turbulent_viscosity) or ``volume_fields_normalization.npz`` "
                    f"next to stats/checkpoint (looked under {stats_path!r})."
                )
            pipe_kw = {**_volume_datapipe_kw(), **dp_user}
            self._datapipe = TransolverDataPipe(
                input_path=None,
                model_type="volume",
                resolution=None,
                surface_factors=None,
                volume_factors=volume_factors,
                scaling_type="mean_std_scaling",
                **pipe_kw,
            )
            self._move_reference_scale_to_device(self._datapipe)
            if self._datapipe.config.scale_invariance and self._datapipe.config.reference_scale is not None:
                self._datapipe.config.reference_scale = self._datapipe.config.reference_scale.to(dev)
            model_kw = dict(DEFAULT_TRANSOLVER_VOLUME_KW)
        else:
            log_inference("transolver", f"Loading surface normalization from {stats_path}")
            surface_factors = load_transolver_surface_factors(stats_path, device)
            pipe_kw = {**_surface_datapipe_kw(), **dp_user}
            self._datapipe = TransolverDataPipe(
                input_path=None,
                model_type="surface",
                resolution=None,
                surface_factors=surface_factors,
                volume_factors=None,
                scaling_type="mean_std_scaling",
                **pipe_kw,
            )
            self._move_reference_scale_to_device(self._datapipe)
            model_kw = dict(DEFAULT_TRANSOLVER_KW)

        self._model = Transolver(**model_kw)
        log_inference("transolver", f"Loading checkpoint from {checkpoint_dir}")
        ckpt_args = {
            "path": str(checkpoint_dir),
            "models": self._model,
        }
        loaded_epoch = load_checkpoint(device=dev, **ckpt_args)
        # load_checkpoint(path=str(checkpoint_dir), models=self._model, device=dev)
        self._model = self._model.to(dev)
        self._model.eval()
        log_inference("transolver", "Checkpoint loaded; model ready for inference.")
        return self

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        if self._datapipe is None:
            raise RuntimeError("TransolverWrapper: call load() first")
        log_inference(
            "transolver",
            f"Reading case inputs (case {case.case_id}): mesh {case.mesh_path}, "
            f"run dir {Path(case.mesh_path).parent}",
        )
        run_dir = Path(case.mesh_path).parent
        run_idx = run_id_from_case_id(case.case_id)
        device = torch.device(self._device)

        if self._inference_mode == "volume":
            data_dict = build_volume_data_dict(
                run_dir=run_dir,
                vtu_path=case.mesh_path,
                device=device,
                air_density=self._air_density,
                stream_velocity=self._stream_velocity,
                run_idx=run_idx,
            )
        else:
            data_dict = build_surface_data_dict(
                run_dir=run_dir,
                vtp_path=case.mesh_path,
                device=device,
                air_density=self._air_density,
                stream_velocity=self._stream_velocity,
                run_idx=run_idx,
            )

        batch = self._datapipe(data_dict)
        return {"batch": batch, "datapipe": self._datapipe}

    def predict(self, model_input: ModelInput) -> RawOutput:
        if self._model is None or self._datapipe is None:
            raise RuntimeError("TransolverWrapper: call load() first")
        log_inference("transolver", "Running forward pass (predicting fields)…")
        batch = model_input["batch"]
        datapipe = model_input["datapipe"]
        dev = batch["embeddings"].device
        N = batch["embeddings"].shape[1]
        batch_res = min(self._batch_resolution, N)
        indices = torch.randperm(N, device=batch["embeddings"].device)
        index_blocks = torch.split(indices, batch_res)
        preds_list = []
        with torch.no_grad():
            for index_block in index_blocks:
                local_embeddings = batch["embeddings"][:, index_block]
                local_fx = batch["fx"][:, index_block]
                local_batch = {
                    "fx": local_fx,
                    "embeddings": local_embeddings,
                    "fields": batch["fields"][:, index_block],
                }
                if "air_density" in batch:
                    local_batch["air_density"] = batch["air_density"]
                if "stream_velocity" in batch:
                    local_batch["stream_velocity"] = batch["stream_velocity"]
                outputs = self._model(fx=local_fx, embedding=local_embeddings)
                preds_list.append(outputs)
            predictions = torch.cat(preds_list, dim=1)
            inverse_indices = torch.empty_like(indices)
            inverse_indices[indices] = torch.arange(N, device=indices.device)
            predictions = predictions[:, inverse_indices]
            # import pdb; pdb.set_trace()
        predictions = predictions.squeeze(0)

        if self._inference_mode == "volume":
            return datapipe.unscale_model_targets(
                predictions,
                air_density=batch.get("air_density"),
                stream_velocity=batch.get("stream_velocity"),
                factor_type="volume",
            )

        predictions = datapipe.unscale_model_targets(
            predictions,
            air_density=batch.get("air_density"),
            stream_velocity=batch.get("stream_velocity"),
            factor_type="surface",
        )
        # predictions = predictions * batch.get("air_density") * (batch.get("stream_velocity") ** 2)
        return predictions

    def decode_outputs(self, raw_output: RawOutput, case: CanonicalCase) -> Predictions:
        pred = raw_output
        if pred.dim() == 3:
            pred = pred.squeeze(0)

        if self._inference_mode == "volume":
            log_inference(
                "transolver",
                "Decoding outputs (velocity + pressure + nut → canonical volume keys)…",
            )
            u = float(self._stream_velocity)
            rho = float(self._air_density)
            dynamic_pressure = rho * (u**2)
            velocity = (pred[:, 0:3] * u).cpu().numpy().astype(np.float32)
            pressure_volume = (pred[:, 3] * dynamic_pressure).cpu().numpy().astype(np.float32)
            turbulent_viscosity = (pred[:, 4] * dynamic_pressure).cpu().numpy().astype(np.float32)
            return build_predictions_dict(
                velocity=velocity,
                pressure_volume=pressure_volume,
                turbulent_viscosity=turbulent_viscosity,
            )

        log_inference("transolver", "Decoding outputs (pressure + WSS to numpy)…")
        dynamic_pressure = self._air_density * (self._stream_velocity ** 2)
        pressure = (pred[:, 0] * dynamic_pressure).cpu().numpy().astype(np.float32)
        wss = (pred[:, 1:4] * dynamic_pressure).cpu().numpy().astype(np.float32)
        return predictions_dict(pressure, wss)

    def _move_reference_scale_to_device(self, dp: TransolverDataPipe) -> None:
        """``reference_scale`` is often created on CPU; mesh tensors live on ``self._device``."""
        if dp.config.scale_invariance and dp.config.reference_scale is not None:
            dev = torch.device(self._device)
            dp.config.reference_scale = dp.config.reference_scale.to(dev)
