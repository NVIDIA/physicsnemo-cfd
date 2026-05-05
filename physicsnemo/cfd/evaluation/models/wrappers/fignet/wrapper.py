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

"""FIGConvUNet (fignet) model wrapper for DrivAerML-style inference."""

from contextlib import nullcontext
from typing import Any, ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import pyvista as pv
import torch

from physicsnemo.models.figconvnet import FIGConvUNet
from physicsnemo.models.figconvnet.components.reductions import REDUCTION_TYPES
from physicsnemo.models.figconvnet.geometries import GridFeaturesMemoryFormat

from physicsnemo.cfd.evaluation.config import _parse_bool
from physicsnemo.cfd.evaluation.common.checkpoint_compat import (
    trusted_torch_load_context,
)
from physicsnemo.cfd.evaluation.common.io import (
    load_global_stats,
    surface_polydata_from_case,
)
from physicsnemo.cfd.evaluation.common.interpolation import interpolate_to_mesh
from physicsnemo.cfd.evaluation.datasets.schema import (
    CanonicalCase,
    InferenceDomain,
    build_predictions_dict,
)
from physicsnemo.cfd.evaluation.models.inference_autocast import cuda_bf16_autocast
from physicsnemo.cfd.evaluation.models.model_registry import (
    CFDModel,
    ModelInput,
    OutputLocation,
    RawOutput,
    Predictions,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference

_DEFAULT_MLP_CHANNELS = (512, 512)

_DEFAULT_RESOLUTION_MEMORY_FORMAT_PAIRS = (
    (GridFeaturesMemoryFormat.b_xc_y_z, (2, 128, 128)),
    (GridFeaturesMemoryFormat.b_yc_x_z, (128, 2, 128)),
    (GridFeaturesMemoryFormat.b_zc_x_y, (128, 128, 2)),
)

_DEFAULT_COMMUNICATION_TYPES = ("sum",)

_DEFAULT_REDUCTIONS = ("mean",)


def _fignet_state_dict_from_checkpoint(checkpoint: object) -> dict[str, Any]:
    """Extract model weights from common checkpoint layouts.

    Training may save under ``model`` (full job dict), ``model_state_dict`` (XMGN-style),
    ``state_dict``, or as a flat :class:`torch.nn.Module` state dict (e.g. ``model_00999.pth``).
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"FIGNet checkpoint must be a dict, got {type(checkpoint).__name__}"
        )
    for key in ("model", "model_state_dict", "state_dict"):
        inner = checkpoint.get(key)
        if isinstance(inner, dict) and inner:
            v0 = next(iter(inner.values()))
            if torch.is_tensor(v0):
                return {k.replace("module.", ""): v for k, v in inner.items()}

    # Flat state_dict file: keys are parameter names, values are tensors.
    sample = [(k, v) for k, v in list(checkpoint.items())[:32] if torch.is_tensor(v)]
    if len(sample) >= 2 and all(isinstance(k, str) for k, _ in sample):
        return {
            k.replace("module.", ""): v
            for k, v in checkpoint.items()
            if torch.is_tensor(v)
        }

    keys_preview = list(checkpoint.keys())[:24]
    raise KeyError(
        "FIGNet checkpoint must contain tensor weights under 'model', 'model_state_dict', "
        f"or 'state_dict', or be a flat state dict. Top-level keys: {keys_preview}"
    )


class FIGConvUNetDrivAerML(FIGConvUNet):
    """FIGConvUNet variant for DrivAerML; parent ``in_channels`` is ``hidden_channels[0]`` (not a separate arg)."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        mlp_channels: Optional[List[int]] = None,
        aabb_max: Tuple[float, float, float] = (2.5, 1.5, 1.0),
        aabb_min: Tuple[float, float, float] = (-2.5, -1.5, -1.0),
        voxel_size: Optional[float] = None,
        resolution_memory_format_pairs: Optional[
            List[Tuple[GridFeaturesMemoryFormat, Tuple[int, int, int]]]
        ] = None,
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = True,
        pos_encode_dim: int = 32,
        communication_types: Optional[List[Literal["mul", "sum"]]] = None,
        to_point_sample_method: Literal["graphconv", "interp"] = "graphconv",
        neighbor_search_type: Literal["knn", "radius"] = "knn",
        knn_k: int = 16,
        reductions: Optional[List[REDUCTION_TYPES]] = None,
        pooling_type: Literal["attention", "max", "mean"] = "max",
        pooling_layers: Optional[List[int]] = None,
    ):
        if mlp_channels is None:
            mlp_channels = list(_DEFAULT_MLP_CHANNELS)
        if resolution_memory_format_pairs is None:
            resolution_memory_format_pairs = [
                (mf, dims) for mf, dims in _DEFAULT_RESOLUTION_MEMORY_FORMAT_PAIRS
            ]
        if communication_types is None:
            communication_types = list(_DEFAULT_COMMUNICATION_TYPES)
        if reductions is None:
            reductions = list(_DEFAULT_REDUCTIONS)
        super().__init__(
            in_channels=hidden_channels[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            mlp_channels=mlp_channels,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            voxel_size=voxel_size,
            resolution_memory_format_pairs=resolution_memory_format_pairs,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            communication_types=communication_types,
            to_point_sample_method=to_point_sample_method,
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
            reductions=reductions,
            pooling_type=pooling_type,
            pooling_layers=pooling_layers,
        )


class FIGNetWrapper(CFDModel):
    """Wrapper for FIGConvUNet: cell-center points, pressure + WSS output.

    **Model kwargs**

    ``cuda_bf16_autocast`` (bool or string, default ``True``)
        CUDA bf16 autocast around the forward pass; set ``false`` for full fp32. Hydra-safe boolean parsing.
    """

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"

    @property
    def output_location(self) -> OutputLocation:
        """See :attr:`CFDModel.output_location` (FIGNet predicts at surface cell centers)."""
        return self.OUTPUT_LOCATION

    def __init__(self) -> None:
        self._model: Optional[FIGConvUNetDrivAerML] = None
        self._stats: Optional[dict] = None
        self._device: str = "cuda:0"
        self._max_points: Optional[int] = None
        self._interpolation_k: int = 4
        self._cuda_bf16_autocast: bool = True

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "FIGNetWrapper":
        """Load FIGNet weights and ``global_stats.json`` normalization tensors onto ``device``."""
        self._device = device
        self._cuda_bf16_autocast = _parse_bool(
            kwargs.pop("cuda_bf16_autocast", None), default=True
        )
        self._max_points = kwargs.get("max_points")
        self._interpolation_k = kwargs.get("interpolation_k", 4)
        log_inference("fignet", f"Loading normalization stats from {stats_path}")
        self._stats = load_global_stats(stats_path, device)
        log_inference("fignet", f"Loading checkpoint from {checkpoint_path}")
        model = FIGConvUNetDrivAerML(
            aabb_max=[2.0, 1.8, 2.6],
            aabb_min=[-2.0, -1.8, -1.5],
            hidden_channels=[16, 16, 16],
            kernel_size=5,
            mlp_channels=[512, 512],
            neighbor_search_type="radius",
            num_down_blocks=1,
            num_levels=2,
            out_channels=4,
            pooling_layers=[2],
            pooling_type="max",
            reductions=["mean"],
            resolution_memory_format_pairs=[
                (GridFeaturesMemoryFormat.b_xc_y_z, [5, 150, 100]),
                (GridFeaturesMemoryFormat.b_yc_x_z, [250, 3, 100]),
                (GridFeaturesMemoryFormat.b_zc_x_y, [250, 150, 2]),
            ],
            use_rel_pos_encode=True,
        )
        with trusted_torch_load_context():
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            state_dict = _fignet_state_dict_from_checkpoint(checkpoint)
            model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        self._model = model
        log_inference("fignet", "Checkpoint loaded; model ready for inference.")
        return self

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        """Read surface mesh, compute normals/sizes, and normalize cell-center coordinates."""
        if self._model is None or self._stats is None:
            raise RuntimeError("FIGNetWrapper: call load() first")
        log_inference(
            "fignet",
            f"Reading mesh (case {case.case_id}): {case.mesh_path}",
        )
        mesh = surface_polydata_from_case(case)
        mesh = mesh.compute_normals()
        mesh = mesh.compute_cell_sizes()
        coords = torch.from_numpy(mesh.cell_centers().points).to(
            self._device, dtype=torch.float32
        )
        coords = coords.unsqueeze(0)
        n_total = coords.shape[1]
        if self._max_points is not None and n_total > self._max_points:
            idx = torch.randperm(n_total)[: self._max_points].to(self._device)
            coords = torch.index_select(coords, 1, idx)
        vertices = (coords - self._stats["mean"]["coordinates"]) / self._stats["std"][
            "coordinates"
        ]
        return {
            "mesh": mesh,
            "vertices": vertices,
            "coords_denorm": coords,
        }

    def predict(self, model_input: ModelInput) -> RawOutput:
        """Run the FIGNet forward pass under the configured autocast context."""
        if self._model is None:
            raise RuntimeError("FIGNetWrapper: call load() first")
        log_inference("fignet", "Running forward pass (predicting fields)…")
        ac_ctx = (
            cuda_bf16_autocast(self._device)
            if self._cuda_bf16_autocast
            else nullcontext()
        )
        with torch.inference_mode():
            with ac_ctx:
                pred, _ = self._model(model_input["vertices"])
        return pred

    def decode_outputs(
        self,
        raw_output: RawOutput,
        case: CanonicalCase,
        model_input: Optional[ModelInput] = None,
    ) -> Predictions:
        """Denormalize predictions and interpolate (when subsampled) back to all surface cells."""
        if self._stats is None:
            raise RuntimeError("FIGNetWrapper: call load() first")
        log_inference(
            "fignet",
            "Decoding outputs (denormalize + interpolate to mesh cells)…",
        )
        mesh = model_input.get("mesh") if model_input else None
        coords_denorm = model_input.get("coords_denorm") if model_input else None

        if self._max_points is not None and (mesh is None or coords_denorm is None):
            raise ValueError(
                "FIGNetWrapper.decode_outputs requires the same model_input dict returned by prepare_inputs "
                "(keys 'mesh', 'coords_denorm') when max_points is set; subsampled field length does not "
                "match reloading the surface mesh alone."
            )
        if mesh is None or coords_denorm is None:
            mesh = surface_polydata_from_case(case)
            mesh = mesh.compute_normals()
            mesh = mesh.compute_cell_sizes()
            coords_denorm = (
                torch.from_numpy(mesh.cell_centers().points)
                .to(self._device, dtype=torch.float32)
                .unsqueeze(0)
            )
        pred = raw_output
        pressure = (
            pred[..., :1] * self._stats["std"]["pressure"]
            + self._stats["mean"]["pressure"]
        )
        wss = (
            pred[..., 1:] * self._stats["std"]["shear_stress"]
            + self._stats["mean"]["shear_stress"]
        )
        target_points = mesh.cell_centers().points
        p_mesh, wss_mesh = interpolate_to_mesh(
            target_points,
            coords_denorm[0].cpu().numpy(),
            pressure[0],
            wss[0],
            k=self._interpolation_k,
        )
        return build_predictions_dict(pressure=p_mesh, shear_stress=wss_mesh)
