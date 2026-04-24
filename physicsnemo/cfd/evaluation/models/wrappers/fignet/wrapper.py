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

from typing import Any, ClassVar, List, Literal, Optional, Tuple, Union

import numpy as np
import pyvista as pv
import torch

from physicsnemo.models.figconvnet import FIGConvUNet
from physicsnemo.models.figconvnet.components.reductions import REDUCTION_TYPES
from physicsnemo.models.figconvnet.geometries import GridFeaturesMemoryFormat

from physicsnemo.cfd.evaluation.common.io import load_global_stats, load_mesh
from physicsnemo.cfd.evaluation.common.interpolation import interpolate_to_mesh
from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase, InferenceDomain, predictions_dict
from physicsnemo.cfd.evaluation.models.model_registry import (
    CFDModel,
    ModelInput,
    OutputLocation,
    RawOutput,
    Predictions,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference


class FIGConvUNetDrivAerML(FIGConvUNet):
    """FIGConvUNet variant for DrivAerML; in_channels remapped to hidden_channels[0]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        mlp_channels: List[int] = [512, 512],
        aabb_max: Tuple[float, float, float] = (2.5, 1.5, 1.0),
        aabb_min: Tuple[float, float, float] = (-2.5, -1.5, -1.0),
        voxel_size: Optional[float] = None,
        resolution_memory_format_pairs: List[
            Tuple[GridFeaturesMemoryFormat, Tuple[int, int, int]]
        ] = [
            (GridFeaturesMemoryFormat.b_xc_y_z, (2, 128, 128)),
            (GridFeaturesMemoryFormat.b_yc_x_z, (128, 2, 128)),
            (GridFeaturesMemoryFormat.b_zc_x_y, (128, 128, 2)),
        ],
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = True,
        pos_encode_dim: int = 32,
        communication_types: List[Literal["mul", "sum"]] = ["sum"],
        to_point_sample_method: Literal["graphconv", "interp"] = "graphconv",
        neighbor_search_type: Literal["knn", "radius"] = "knn",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
        pooling_type: Literal["attention", "max", "mean"] = "max",
        pooling_layers: Optional[List[int]] = None,
    ):
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
    """Wrapper for FIGConvUNet: cell-center points, pressure + WSS output."""

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "cell"

    @property
    def output_location(self) -> OutputLocation:
        return self.OUTPUT_LOCATION

    def __init__(self) -> None:
        self._model: Optional[FIGConvUNetDrivAerML] = None
        self._stats: Optional[dict] = None
        self._device: str = "cuda:0"
        self._max_points: Optional[int] = None
        self._interpolation_k: int = 4

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "FIGNetWrapper":
        self._device = device
        self._max_points = kwargs.get("max_points")
        self._interpolation_k = kwargs.get("interpolation_k", 4)
        log_inference("fignet", f"Loading normalization stats from {stats_path}")
        self._stats = load_global_stats(stats_path, device)
        log_inference("fignet", f"Loading checkpoint from {checkpoint_path}")
        model = FIGConvUNetDrivAerML(
            aabb_max=[2.0, 1.8, 2.6],
            aabb_min=[-2.0, -1.8, -1.5],
            hidden_channels=[16, 16, 16],
            in_channels=1,
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
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        model = model.to(device)
        model.eval()
        self._model = model
        log_inference("fignet", "Checkpoint loaded; model ready for inference.")
        return self

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        if self._model is None or self._stats is None:
            raise RuntimeError("FIGNetWrapper: call load() first")
        log_inference(
            "fignet",
            f"Reading mesh (case {case.case_id}): {case.mesh_path}",
        )
        mesh = load_mesh(case.mesh_path)
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
        self._last_mesh = mesh
        self._last_coords_denorm = coords
        return {
            "mesh": mesh,
            "vertices": vertices,
            "coords_denorm": coords,
        }

    def predict(self, model_input: ModelInput) -> RawOutput:
        if self._model is None:
            raise RuntimeError("FIGNetWrapper: call load() first")
        log_inference("fignet", "Running forward pass (predicting fields)…")
        with torch.inference_mode():
            pred, _ = self._model(model_input["vertices"])
        return pred

    def decode_outputs(self, raw_output: RawOutput, case: CanonicalCase) -> Predictions:
        if self._stats is None:
            raise RuntimeError("FIGNetWrapper: call load() first")
        log_inference(
            "fignet",
            "Decoding outputs (denormalize + interpolate to mesh cells)…",
        )
        # model_input was prepare_inputs output; we need mesh and coords_denorm from there.
        # We don't have it in decode_outputs - we only have case. So we must either pass
        # model_input through to decode_outputs or re-load mesh in decode_outputs.
        # Plan says: decode_outputs(raw_output, case). So we need mesh in case or re-load.
        # CanonicalCase has mesh_path, not mesh object. So we re-load mesh here to get
        # cell centers for interpolation target. We have raw_output (pred) and stats;
        # we need pred_coords (where pred was computed) - that's not in case.
        # So the design has a gap: decode_outputs doesn't receive the model_input that
        # had coords_denorm. Options: (1) Change API to decode_outputs(raw_output, case, model_input)
        # or (2) Have prepare_inputs return a ModelInput that we pass to predict and then
        # the engine passes the same model_input to decode_outputs. Plan says decode_outputs(raw_output, case).
        # So we need to stash coords_denorm and mesh somewhere. Easiest: have decode_outputs
        # accept optional model_input (for interpolation). Or stash in raw_output by
        # having predict return a dict { "pred": tensor, "model_input": model_input }.
        # That would require the engine to pass model_input to decode_outputs. Let me
        # re-read the plan. "decode_outputs(raw_output: RawOutput, case: CanonicalCase) -> Predictions"
        # So we can't change the signature. Then the only way is to re-run prepare_inputs in
        # decode_outputs to get mesh and coords_denorm again (wasteful but correct), or
        # store the last model_input on self (stateful). I'll store last model_input on self
        # so decode_outputs can use it (we're single-threaded per model).
        mesh = getattr(self, "_last_mesh", None)
        coords_denorm = getattr(self, "_last_coords_denorm", None)
        if mesh is None or coords_denorm is None:
            # Fallback: reload from case (will not have subsampled coords; wrong for subsampled run)
            mesh = load_mesh(case.mesh_path)
            mesh = mesh.compute_normals()
            mesh = mesh.compute_cell_sizes()
            coords_denorm = torch.from_numpy(mesh.cell_centers().points).to(
                self._device, dtype=torch.float32
            ).unsqueeze(0)
        pred = raw_output
        pressure = pred[..., :1] * self._stats["std"]["pressure"] + self._stats["mean"]["pressure"]
        wss = pred[..., 1:] * self._stats["std"]["shear_stress"] + self._stats["mean"]["shear_stress"]
        target_points = mesh.cell_centers().points
        p_mesh, wss_mesh = interpolate_to_mesh(
            target_points,
            coords_denorm[0].cpu().numpy(),
            pressure[0],
            wss[0],
            k=self._interpolation_k,
        )
        return predictions_dict(p_mesh, wss_mesh)
