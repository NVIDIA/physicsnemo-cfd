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

"""GeoTransolver model wrapper for surface or volume inference (TransolverDataPipe + VTK).

Surface: cell-centered pressure + WSS on boundary VTP (``inference_domain: surface`` or default).

Volume: velocity + pressure + turbulent viscosity on volume VTU (``inference_domain: volume``),
aligned with ``examples/cfd/external_aerodynamics/transformer_models/src/inference_on_vtk.py``.
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
    from physicsnemo.experimental.models.geotransolver import GeoTransolver
    from physicsnemo.utils import load_checkpoint

    _PHYSICSNEMO_AVAILABLE = True
except ImportError:
    _PHYSICSNEMO_AVAILABLE = False


# Default GeoTransolver surface config (from geotransolver_surface + model/geotransolver.yaml)
DEFAULT_GEOTRANSOLVER_KW = dict(
    functional_dim=6,
    global_dim=2,
    geometry_dim=3,
    out_dim=4,
    n_layers=20,
    n_hidden=256,
    dropout=0.0,
    n_head=8,
    act="gelu",
    mlp_ratio=2,
    slice_num=128,
    use_te=False,
    plus=False,
    include_local_features=True,
    radii=[0.01, 0.05, 0.25, 1.0, 2.5, 5.0],
    neighbors_in_radius=[4, 8, 16, 64, 128, 256],
    n_hidden_local=32,
)

# Volume training defaults (geotransolver_volume.yaml)
DEFAULT_GEOTRANSOLVER_VOLUME_KW = {
    **DEFAULT_GEOTRANSOLVER_KW,
    "functional_dim": 7,
    "out_dim": 5,
}

def _global_fx_to_bnc(fx: torch.Tensor) -> torch.Tensor:
    """GeoTransolver requires ``global_embedding`` shape (B, N_g, C_g).

    With ``broadcast_global_features=False``, ``TransolverDataPipe`` may stack ``fx`` as
    (1, 1, C) before ``__call__`` adds another batch dimension, yielding (1, 1, 1, C).
    Squeeze singleton middle dims until 3D.
    """
    out = fx
    while out.ndim > 3:
        for d in range(1, out.ndim - 1):
            if out.shape[d] == 1:
                out = out.squeeze(d)
                break
        else:
            raise ValueError(
                "Cannot reshape global ``fx`` to 3D for GeoTransolver; "
                f"shape={tuple(fx.shape)}"
            )
    return out


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


def _surface_datapipe_static_kw() -> dict[str, Any]:
    return dict(
        include_normals=True,
        include_sdf=False,
        broadcast_global_features=False,
        include_geometry=True,
        translational_invariance=True,
        scale_invariance=True,
        reference_scale=[12.0, 4.5, 3.25]
    )


def _volume_datapipe_static_kw() -> dict[str, Any]:
    """Defaults aligned with ``data/core.yaml`` + ``geotransolver_volume.yaml``."""
    return dict(
        include_normals=True,
        include_sdf=True,
        translational_invariance=True,
        scale_invariance=True,
        reference_scale=[12.0, 4.5, 3.25],
        broadcast_global_features=False,
        include_geometry=True,
        return_mesh_features=False,
    )


class GeoTransolverWrapper(CFDModel):
    """GeoTransolver: VTP+STL (surface) or VTU+STL (volume) via ``inference_domain`` in load kwargs."""

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"

    @property
    def output_location(self) -> OutputLocation:
        return self.OUTPUT_LOCATION

    def __init__(self) -> None:
        self._model: Optional[GeoTransolver] = None
        self._datapipe: Optional[TransolverDataPipe] = None
        self._datapipe_geometry_effective: Optional[int] = None
        self._surface_factors: Any = None
        self._volume_factors: Any = None
        self._geometry_sampling_requested: int = 300_000
        self._datapipe_resolution: int = 100_000
        self._device: str = "cuda:0"
        self._air_density: float = 1.205
        self._stream_velocity: float = 30.0
        self._batch_resolution: int = 2048
        self._inference_mode: Literal["surface", "volume"] = "surface"
        self._datapipe_user_kw: dict[str, Any] = {}

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "GeoTransolverWrapper":
        if not _PHYSICSNEMO_AVAILABLE:
            raise RuntimeError(
                "GeoTransolver wrapper requires physicsnemo (GeoTransolver, TransolverDataPipe, load_checkpoint)."
            )
        kw = dict(kwargs)
        dom = kw.pop("inference_domain", None)
        self._inference_mode = dom if dom in ("surface", "volume") else "surface"

        self._device = device
        self._air_density = float(kw.get("air_density", 1.205))
        self._stream_velocity = float(kw.get("stream_velocity", 30.0))
        self._batch_resolution = int(kw.get("batch_resolution", 2048))

        checkpoint_dir = Path(checkpoint_path)
        if checkpoint_dir.is_file():
            checkpoint_dir = checkpoint_dir.parent

        dp_user = {k: kw.pop(k) for k in list(kw.keys()) if k in _DATAPIPE_KEYS}

        if self._inference_mode == "volume":
            log_inference("geotransolver", f"Loading volume normalization from {stats_path}")
            self._volume_factors = load_transolver_volume_factors(stats_path, device)
            if self._volume_factors is None:
                raise FileNotFoundError(
                    "Volume inference requires ``global_stats.json`` (with velocity, "
                    "pressure, turbulent_viscosity) or ``volume_fields_normalization.npz`` "
                    f"next to stats/checkpoint (looked under {stats_path!r})."
                )
            self._surface_factors = None
            model_kw = dict(DEFAULT_GEOTRANSOLVER_VOLUME_KW)
        else:
            log_inference("geotransolver", f"Loading surface normalization from {stats_path}")
            self._surface_factors = load_transolver_surface_factors(stats_path, device)
            self._volume_factors = None
            model_kw = dict(DEFAULT_GEOTRANSOLVER_KW)

        self._geometry_sampling_requested = int(kw.get("geometry_sampling", 300_000))
        self._datapipe_resolution = int(kw.get("resolution", 100_000))

        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        dev = torch.device(device)

        self._datapipe = None
        self._datapipe_geometry_effective = None
        self._datapipe_user_kw = dp_user

        self._model = GeoTransolver(**model_kw)
        ckpt_args = {
            "path": str(checkpoint_dir),
            "models": self._model,
        }
        with trusted_torch_load_context():
            _ = load_checkpoint(device=dev, **ckpt_args)
        self._model = self._model.to(dev)
        self._model.eval()
        return self

    def _move_reference_scale_to_device(self, dp: TransolverDataPipe) -> None:
        """``reference_scale`` is often created on CPU; mesh tensors live on ``self._device``."""
        if dp.config.scale_invariance and dp.config.reference_scale is not None:
            dev = torch.device(self._device)
            dp.config.reference_scale = dp.config.reference_scale.to(dev)

    def _make_datapipe_surface(self, geometry_sampling: int) -> TransolverDataPipe:
        dp = TransolverDataPipe(
            input_path=None,
            model_type="surface",
            resolution=None,
            surface_factors=self._surface_factors,
            volume_factors=None,
            scaling_type="mean_std_scaling",
            return_mesh_features=True,
            geometry_sampling=geometry_sampling,
            **_surface_datapipe_static_kw(),
        )
        self._move_reference_scale_to_device(dp)
        return dp

    def _make_datapipe_volume(
        self, geometry_sampling: int, user_kw: dict[str, Any]
    ) -> TransolverDataPipe:
        merged = {**_volume_datapipe_static_kw(), **user_kw}
        merged["geometry_sampling"] = geometry_sampling
        dp = TransolverDataPipe(
            input_path=None,
            model_type="volume",
            resolution=None,
            surface_factors=None,
            volume_factors=self._volume_factors,
            scaling_type="mean_std_scaling",
            **merged,
        )
        self._move_reference_scale_to_device(dp)
        return dp

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        if self._model is None:
            raise RuntimeError("GeoTransolverWrapper: call load() first")
        log_inference(
            "geotransolver",
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
            n_stl = int(data_dict["stl_coordinates"].shape[0])
            safe_geo = max(1, min(self._geometry_sampling_requested, n_stl))
            user_kw = self._datapipe_user_kw
            if self._datapipe is None or self._datapipe_geometry_effective != safe_geo:
                self._datapipe = self._make_datapipe_volume(safe_geo, user_kw)
                self._datapipe_geometry_effective = safe_geo
        else:
            data_dict = build_surface_data_dict(
                run_dir=run_dir,
                vtp_path=case.mesh_path,
                device=device,
                air_density=self._air_density,
                stream_velocity=self._stream_velocity,
                run_idx=run_idx,
            )
            n_stl = int(data_dict["stl_coordinates"].shape[0])
            n_surf = int(data_dict["surface_mesh_centers"].shape[0])
            safe_geo = min(self._geometry_sampling_requested, n_stl, n_surf)
            safe_geo = max(1, safe_geo)
            if self._datapipe is None or self._datapipe_geometry_effective != safe_geo:
                self._datapipe = self._make_datapipe_surface(safe_geo)
                self._datapipe_geometry_effective = safe_geo

        batch = self._datapipe(data_dict)
        return {"batch": batch, "datapipe": self._datapipe}

    def predict(self, model_input: ModelInput) -> RawOutput:
        if self._model is None or self._datapipe is None:
            raise RuntimeError("GeoTransolverWrapper: call load() first")
        log_inference("geotransolver", "Running forward pass (predicting fields)…")
        batch = model_input["batch"]
        datapipe = model_input["datapipe"]
        fx_bn_c = _global_fx_to_bnc(batch["fx"])
        N = batch["embeddings"].shape[1]
        batch_res = min(self._batch_resolution, N)
        indices = torch.randperm(N, device=batch["embeddings"].device)
        index_blocks = torch.split(indices, batch_res)
        preds_list = []
        use_full_fx = "geometry" in batch

        with torch.no_grad():
            for index_block in index_blocks:
                local_embeddings = batch["embeddings"][:, index_block]
                local_fields = batch["fields"][:, index_block]
                local_fx = fx_bn_c if use_full_fx else fx_bn_c[:, index_block]
                local_batch = {
                    "fx": local_fx,
                    "embeddings": local_embeddings,
                    "fields": local_fields,
                }
                if "geometry" in batch:
                    local_batch["geometry"] = batch["geometry"]
                if "air_density" in batch:
                    local_batch["air_density"] = batch["air_density"]
                if "stream_velocity" in batch:
                    local_batch["stream_velocity"] = batch["stream_velocity"]
                local_positions = local_embeddings[:, :, :3]
                outputs = self._model(
                    local_embedding=local_embeddings,
                    local_positions=local_positions,
                    global_embedding=local_fx,
                    geometry=local_batch.get("geometry"),
                )
                preds_list.append(outputs)
            predictions = torch.cat(preds_list, dim=1)
            inverse_indices = torch.empty_like(indices)
            inverse_indices[indices] = torch.arange(N, device=indices.device)
            predictions = predictions[:, inverse_indices]
        predictions = predictions.squeeze(0)

        if self._inference_mode == "volume":
            predictions = datapipe.unscale_model_targets(
                predictions,
                air_density=batch.get("air_density"),
                stream_velocity=batch.get("stream_velocity"),
                factor_type="volume",
            )
            return predictions

        predictions = datapipe.unscale_model_targets(
            predictions,
            air_density=batch.get("air_density"),
            stream_velocity=batch.get("stream_velocity"),
            factor_type="surface",
        )
        return predictions

    def decode_outputs(self, raw_output: RawOutput, case: CanonicalCase) -> Predictions:
        pred = raw_output
        if pred.dim() == 3:
            pred = pred.squeeze(0)

        if self._inference_mode == "volume":
            log_inference(
                "geotransolver",
                "Decoding outputs (velocity + pressure + nut → canonical volume keys)…",
            )
            u = float(self._stream_velocity)
            rho = float(self._air_density)
            dynamic_pressure = rho * (u**2)
            velocity = (pred[:, 0:3] * u).cpu().numpy().astype(np.float32)
            pressure = (pred[:, 3] * dynamic_pressure).cpu().numpy().astype(np.float32)
            turbulent_viscosity = (pred[:, 4] * dynamic_pressure).cpu().numpy().astype(np.float32)
            return build_predictions_dict(
                velocity=velocity,
                pressure=pressure,
                turbulent_viscosity=turbulent_viscosity,
            )

        log_inference("geotransolver", "Decoding outputs (pressure + WSS to numpy)…")
        dynamic_pressure = self._air_density * (self._stream_velocity ** 2)
        pressure = (pred[:, 0] * dynamic_pressure).cpu().numpy().astype(np.float32)
        wss = (pred[:, 1:4] * dynamic_pressure).cpu().numpy().astype(np.float32)
        return predictions_dict(pressure, wss)
