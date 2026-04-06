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

"""MeshGraphNet (xmgn) model wrapper for DrivAerML-style inference."""

import os
os.environ.setdefault("PHYSICSNEMO_FORCE_TE", "False")

from typing import Any, ClassVar, Optional

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data as PyGData

from physicsnemo.models.meshgraphnet import MeshGraphNet

from physicsnemo.cfd.evaluation.common.io import load_global_stats, load_mesh
from physicsnemo.cfd.evaluation.common.interpolation import interpolate_to_mesh
from physicsnemo.cfd.evaluation.datasets.schema import CanonicalCase, InferenceDomain, predictions_dict
from physicsnemo.cfd.evaluation.inference.model_registry import (
    CFDModel,
    ModelInput,
    OutputLocation,
    RawOutput,
    Predictions,
)
from physicsnemo.cfd.evaluation.inference.progress import log_inference


def _build_pyg_graph(
    points: np.ndarray,
    normals: np.ndarray,
    node_degree: int,
) -> tuple[PyGData, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_points = len(points)
    nbrs = NearestNeighbors(n_neighbors=node_degree + 1, algorithm="ball_tree").fit(points)
    _, indices = nbrs.kneighbors(points)
    src = np.repeat(np.arange(n_points), node_degree)
    dst = indices[:, 1:].flatten()
    src_bi = np.concatenate([src, dst])
    dst_bi = np.concatenate([dst, src])
    edges = np.unique(np.stack([src_bi, dst_bi], axis=0), axis=1)
    self_loops = np.stack([np.arange(n_points), np.arange(n_points)], axis=0)
    edge_index = torch.tensor(np.concatenate([edges, self_loops], axis=1), dtype=torch.long)
    coordinates = torch.tensor(points, dtype=torch.float32)
    normals_t = torch.tensor(normals, dtype=torch.float32)
    disp = coordinates[edge_index[0]] - coordinates[edge_index[1]]
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
    edge_features = torch.cat((disp, disp_norm), dim=-1)
    graph = PyGData(edge_index=edge_index)
    return graph, coordinates, normals_t, edge_features


def _prepare_node_features(
    coords: torch.Tensor,
    normals: torch.Tensor,
    stats: dict,
    device: str,
) -> torch.Tensor:
    coords = coords.to(device)
    normals = normals.to(device)
    coords_norm = (coords - stats["mean"]["coordinates"]) / stats["std"]["coordinates"]
    normals_norm = (normals - stats["mean"]["normals"]) / stats["std"]["normals"]
    ndata = torch.cat([
        coords_norm,
        normals_norm,
        torch.sin(2 * np.pi * coords_norm),
        torch.cos(2 * np.pi * coords_norm),
        torch.sin(4 * np.pi * coords_norm),
        torch.cos(4 * np.pi * coords_norm),
        torch.sin(8 * np.pi * coords_norm),
        torch.cos(8 * np.pi * coords_norm),
    ], dim=1)
    return ndata


class XMGNWrapper(CFDModel):
    """Wrapper for MeshGraphNet: mesh points + PyG graph, pressure + WSS output."""

    INFERENCE_DOMAIN: ClassVar[InferenceDomain] = "surface"
    OUTPUT_LOCATION: ClassVar[OutputLocation] = "point"

    @property
    def output_location(self) -> OutputLocation:
        return self.OUTPUT_LOCATION

    def __init__(self) -> None:
        self._model: Optional[MeshGraphNet] = None
        self._stats: Optional[dict] = None
        self._device: str = "cuda:0"
        self._max_points: Optional[int] = None
        self._node_degree: int = 6
        self._interpolation_k: int = 5

    def load(
        self,
        checkpoint_path: str,
        stats_path: str,
        device: str,
        **kwargs: Any,
    ) -> "XMGNWrapper":
        self._device = device
        self._max_points = kwargs.get("max_points")
        self._node_degree = kwargs.get("node_degree", 6)
        self._interpolation_k = kwargs.get("interpolation_k", 5)
        log_inference("xmgn", f"Loading normalization stats from {stats_path}")
        self._stats = load_global_stats(stats_path, device)
        log_inference("xmgn", f"Loading checkpoint from {checkpoint_path}")
        model = MeshGraphNet(
            input_dim_nodes=24,
            input_dim_edges=4,
            output_dim=4,
            processor_size=15,
            aggregation="sum",
            hidden_dim_node_encoder=512,
            hidden_dim_edge_encoder=512,
            hidden_dim_node_decoder=512,
            mlp_activation_fn="silu",
            do_concat_trick=True,
            num_processor_checkpoint_segments=3,
            norm_type="LayerNorm",
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model
        log_inference("xmgn", "Checkpoint loaded; model ready for inference.")
        return self

    def prepare_inputs(self, case: CanonicalCase) -> ModelInput:
        if self._model is None or self._stats is None:
            raise RuntimeError("XMGNWrapper: call load() first")
        log_inference(
            "xmgn",
            f"Reading mesh (case {case.case_id}): {case.mesh_path}",
        )
        mesh = load_mesh(case.mesh_path)
        if "Normals" not in mesh.point_data:
            mesh = mesh.compute_normals(point_normals=True, cell_normals=False, flip_normals=True)
        points = np.array(mesh.points)
        normals = np.array(mesh.point_data["Normals"])
        n_total = len(points)
        if self._max_points is not None and n_total > self._max_points:
            idx = np.random.choice(n_total, self._max_points, replace=False)
            points = points[idx]
            normals = normals[idx]
        graph, coords, normals_t, edge_features = _build_pyg_graph(
            points, normals, self._node_degree
        )
        ndata = _prepare_node_features(coords, normals_t, self._stats, self._device)
        self._last_mesh = mesh
        self._last_pred_coords = points
        return {
            "graph": graph,
            "ndata": ndata,
            "edata": edge_features,
        }

    def predict(self, model_input: ModelInput) -> RawOutput:
        if self._model is None or self._stats is None:
            raise RuntimeError("XMGNWrapper: call load() first")
        log_inference("xmgn", "Running forward pass (predicting fields)…")
        graph = model_input["graph"].to(self._device)
        edata = model_input["edata"].to(self._device)
        edata_norm = (edata - self._stats["mean"]["x"]) / self._stats["std"]["x"]
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                pred = self._model(model_input["ndata"], edata_norm, graph)
        return pred

    def decode_outputs(self, raw_output: RawOutput, case: CanonicalCase) -> Predictions:
        if self._stats is None:
            raise RuntimeError("XMGNWrapper: call load() first")
        log_inference(
            "xmgn",
            "Decoding outputs (denormalize + interpolate to mesh points)…",
        )
        pred = raw_output
        pressure = pred[:, 0:1] * self._stats["std"]["pressure"] + self._stats["mean"]["pressure"]
        wss = pred[:, 1:] * self._stats["std"]["shear_stress"] + self._stats["mean"]["shear_stress"]
        mesh = getattr(self, "_last_mesh", None)
        pred_coords = getattr(self, "_last_pred_coords", None)
        if mesh is None or pred_coords is None:
            mesh = load_mesh(case.mesh_path)
            pred_coords = np.array(mesh.points)
        target_points = np.array(mesh.points)
        p_mesh, wss_mesh = interpolate_to_mesh(
            target_points,
            pred_coords,
            pressure,
            wss,
            k=self._interpolation_k,
        )
        return predictions_dict(p_mesh, wss_mesh)
