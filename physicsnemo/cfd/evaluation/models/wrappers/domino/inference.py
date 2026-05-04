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

"""DoMINO surface inference helpers (aligned with physicsnemo domino ``src/test.py``)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
import pyvista as pv
import torch
import vtk
from omegaconf import DictConfig

from physicsnemo.models.domino.utils import (
    calculate_center_of_mass,
    create_grid,
    normalize,
    unnormalize,
    unstandardize,
)
from physicsnemo.models.domino.geometry_rep import scale_sdf
from physicsnemo.models.domino.utils.vtk_file_utils import (
    get_fields,
    get_node_to_elem,
)
from physicsnemo.nn.functional import knn, signed_distance_field

from physicsnemo.cfd.evaluation.datasets.schema import build_predictions_dict
from physicsnemo.cfd.evaluation.models.common_wrapper_utils.vtk_datapipe_io import (
    _find_stl_in_dir,
)
from physicsnemo.cfd.postprocessing_tools.metrics.l2_errors import (
    triangulate_surface_mesh,
)


def domino_count_output_features(cfg: DictConfig) -> tuple[int | None, int | None, int]:
    """Match ``test.py`` variable counting for ``DoMINO`` construction."""
    model_type = cfg.model.model_type
    num_vol_vars = None
    num_surf_vars = None
    if model_type in ("volume", "combined"):
        names = list(cfg.variables.volume.solution.keys())
        num_vol_vars = 0
        for j in names:
            num_vol_vars += 3 if cfg.variables.volume.solution[j] == "vector" else 1
    if model_type in ("surface", "combined"):
        names = list(cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in names:
            num_surf_vars += 3 if cfg.variables.surface.solution[j] == "vector" else 1
    global_features = 0
    for param in cfg.variables.global_parameters.keys():
        g = cfg.variables.global_parameters[param]
        if g.type == "vector":
            global_features += len(g.reference)
        else:
            global_features += 1
    return num_vol_vars, num_surf_vars, global_features


def _volume_solution_field_kind(
    name: str,
    typ: str,
) -> Literal["velocity_vector", "pressure_scalar", "nut_scalar", "other"]:
    """Classify a ``variables.volume.solution`` field for canonical keys and physical scaling."""
    nl = name.lower()
    if typ == "vector":
        if nl.startswith("u") or "velocity" in nl:
            return "velocity_vector"
        return "other"
    if "nut" in nl or "turbulent" in nl or "viscosity" in nl:
        return "nut_scalar"
    if "pmean" in nl or nl.startswith("p_") or nl == "p":
        return "pressure_scalar"
    return "other"


def build_domin_surface_datadict(
    cfg: DictConfig,
    run_dir: Path,
    vtp_path: str,
    tag: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build batched (leading dim 1) surface ``data_dict`` like ``test.py`` (surface-only)."""
    if cfg.model.model_type != "surface":
        raise ValueError(
            f"build_domin_surface_datadict requires model.model_type 'surface', got {cfg.model.model_type!r}"
        )

    surface_variable_names = list(cfg.variables.surface.solution.keys())
    stl_path = _find_stl_in_dir(run_dir, tag)

    reader = pv.get_reader(str(stl_path))
    mesh_stl = triangulate_surface_mesh(reader.read())
    stl_vertices = mesh_stl.points
    stl_faces = np.asarray(mesh_stl.regular_faces, dtype=np.int64)
    mesh_indices_flattened = stl_faces.flatten()
    length_scale = np.array(
        np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0)),
        dtype=np.float32,
    )
    length_scale = torch.from_numpy(length_scale).to(torch.float32).to(device)
    stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
    stl_sizes = np.array(stl_sizes.cell_data["Area"], dtype=np.float32)
    stl_centers = np.array(mesh_stl.cell_centers().points, dtype=np.float32)

    stl_vertices_t = torch.from_numpy(stl_vertices).to(torch.float32).to(device)
    stl_sizes_t = torch.from_numpy(stl_sizes).to(torch.float32).to(device)
    stl_centers_t = torch.from_numpy(stl_centers).to(torch.float32).to(device)
    mesh_indices_flattened_t = (
        torch.from_numpy(mesh_indices_flattened).to(torch.int32).to(device)
    )

    center_of_mass = calculate_center_of_mass(stl_centers_t, stl_sizes_t)

    s_max = torch.from_numpy(np.asarray(cfg.data.bounding_box_surface.max)).to(
        torch.float32
    ).to(device)
    s_min = torch.from_numpy(np.asarray(cfg.data.bounding_box_surface.min)).to(
        torch.float32
    ).to(device)

    nx, ny, nz = cfg.model.interp_res
    surf_grid = create_grid(
        s_max, s_min, torch.from_numpy(np.asarray([nx, ny, nz])).to(device)
    )

    normed_stl_vertices_cp = normalize(stl_vertices_t, s_max, s_min)
    surf_grid_normed = normalize(surf_grid, s_max, s_min)

    sdf_surf_grid, _ = signed_distance_field(
        normed_stl_vertices_cp,
        mesh_indices_flattened_t,
        surf_grid_normed,
        use_sign_winding_number=True,
    )
    surf_grid_max_min = torch.stack([s_min, s_max])

    global_params_reference: dict[str, Any] = {
        name: cfg.variables.global_parameters[name]["reference"]
        for name in cfg.variables.global_parameters.keys()
    }
    global_params_types = {
        name: cfg.variables.global_parameters[name]["type"]
        for name in cfg.variables.global_parameters.keys()
    }
    stream_velocity = global_params_reference["inlet_velocity"][0]
    air_density = global_params_reference["air_density"]

    global_params_reference_list: list[float] = []
    for name, typ in global_params_types.items():
        if typ == "vector":
            global_params_reference_list.extend(global_params_reference[name])
        elif typ == "scalar":
            global_params_reference_list.append(global_params_reference[name])
        else:
            raise ValueError(f"Global parameter {name} type not supported")
    global_params_reference_t = torch.from_numpy(
        np.array(global_params_reference_list, dtype=np.float32)
    ).to(device)

    global_params_values_list: list[float] = []
    for key in global_params_types.keys():
        if key == "inlet_velocity":
            global_params_values_list.append(stream_velocity)
        elif key == "air_density":
            global_params_values_list.append(air_density)
        else:
            raise ValueError(f"Global parameter {key} not supported for this recipe")
    global_params_values_t = torch.from_numpy(
        np.array(global_params_values_list, dtype=np.float32)
    ).to(device)

    vtk_reader = vtk.vtkXMLPolyDataReader()
    vtk_reader.SetFileName(vtp_path)
    vtk_reader.Update()
    polydata_surf = vtk_reader.GetOutput()
    celldata_all = get_node_to_elem(polydata_surf)
    celldata = celldata_all.GetCellData()
    sf_list = get_fields(celldata, surface_variable_names)
    surface_fields = np.concatenate(sf_list, axis=-1)  # (n_cells, n_components)

    mesh = pv.PolyData(polydata_surf)
    surface_coordinates = np.array(mesh.cell_centers().points, dtype=np.float32)
    surface_normals = np.array(mesh.cell_normals, dtype=np.float32)
    surface_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    surface_sizes = np.array(surface_sizes.cell_data["Area"], dtype=np.float32)
    surface_normals = surface_normals / (
        np.linalg.norm(surface_normals, axis=1)[:, np.newaxis] + 1e-8
    )

    surface_coordinates = (
        torch.from_numpy(surface_coordinates).to(torch.float32).to(device)
    )
    surface_normals = torch.from_numpy(surface_normals).to(torch.float32).to(device)
    surface_sizes = torch.from_numpy(surface_sizes).to(torch.float32).to(device)
    surface_fields = torch.from_numpy(surface_fields).to(torch.float32).to(device)

    if cfg.model.num_neighbors_surface > 1:
        ii, _dd = knn(
            points=surface_coordinates,
            queries=surface_coordinates,
            k=cfg.model.num_neighbors_surface,
        )
        surface_neighbors = surface_coordinates[ii][:, 1:]
        surface_neighbors_normals = surface_normals[ii][:, 1:]
        surface_neighbors_sizes = surface_sizes[ii][:, 1:]
    else:
        surface_neighbors = surface_coordinates
        surface_neighbors_normals = surface_normals
        surface_neighbors_sizes = surface_sizes

    if cfg.data.normalize_coordinates:
        surface_coordinates = normalize(surface_coordinates, s_max, s_min)
        surf_grid = normalize(surf_grid, s_max, s_min)
        center_of_mass_normalized = normalize(center_of_mass, s_max, s_min)
        surface_neighbors = normalize(surface_neighbors, s_max, s_min)
    else:
        center_of_mass_normalized = center_of_mass
    pos_surface_center_of_mass = surface_coordinates - center_of_mass_normalized

    geom_centers = stl_vertices_t

    data_dict: dict[str, torch.Tensor] = {
        "pos_surface_center_of_mass": pos_surface_center_of_mass,
        "geometry_coordinates": geom_centers,
        "surf_grid": surf_grid,
        "sdf_surf_grid": sdf_surf_grid,
        "surface_mesh_centers": surface_coordinates,
        "surface_mesh_neighbors": surface_neighbors,
        "surface_normals": surface_normals,
        "surface_neighbors_normals": surface_neighbors_normals,
        "surface_areas": surface_sizes,
        "surface_neighbors_areas": surface_neighbors_sizes,
        "surface_fields": surface_fields,
        "surface_min_max": surf_grid_max_min,
        "length_scale": length_scale,
        "global_params_values": torch.unsqueeze(global_params_values_t, -1),
        "global_params_reference": torch.unsqueeze(global_params_reference_t, -1),
    }
    return {k: torch.unsqueeze(v, 0) for k, v in data_dict.items()}


def build_domin_volume_datadict(
    cfg: DictConfig,
    run_dir: Path,
    vtu_path: str,
    tag: int,
    device: torch.device,
    *,
    reference_mesh: pv.DataSet | None = None,
) -> dict[str, torch.Tensor]:
    """Build batched (leading dim 1) volume ``data_dict`` like domino ``test.py`` (volume-only).

    ``reference_mesh`` (e.g. :attr:`CanonicalCase.reference_geometry`) skips a redundant
    ``pv.read(vtu_path)`` when the adapter already loaded the VTU. The model is evaluated at
    mesh points (matches the convention of all benchmark volume wrappers and point-located GT).
    """
    if cfg.model.model_type != "volume":
        raise ValueError(
            f"build_domin_volume_datadict requires model.model_type 'volume', got {cfg.model.model_type!r}"
        )

    volume_variable_names = list(cfg.variables.volume.solution.keys())
    stl_path = _find_stl_in_dir(run_dir, tag)

    reader = pv.get_reader(str(stl_path))
    mesh_stl = triangulate_surface_mesh(reader.read())
    stl_vertices = mesh_stl.points
    stl_faces = np.asarray(mesh_stl.regular_faces, dtype=np.int64)
    mesh_indices_flattened = stl_faces.flatten()
    length_scale = np.array(
        np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0)),
        dtype=np.float32,
    )
    length_scale = torch.from_numpy(length_scale).to(torch.float32).to(device)
    stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
    stl_sizes = np.array(stl_sizes.cell_data["Area"], dtype=np.float32)
    stl_centers = np.array(mesh_stl.cell_centers().points, dtype=np.float32)

    stl_vertices_t = torch.from_numpy(stl_vertices).to(torch.float32).to(device)
    stl_sizes_t = torch.from_numpy(stl_sizes).to(torch.float32).to(device)
    stl_centers_t = torch.from_numpy(stl_centers).to(torch.float32).to(device)
    mesh_indices_flattened_t = (
        torch.from_numpy(mesh_indices_flattened).to(torch.int32).to(device)
    )

    center_of_mass = calculate_center_of_mass(stl_centers_t, stl_sizes_t)

    s_max = torch.from_numpy(np.asarray(cfg.data.bounding_box_surface.max)).to(
        torch.float32
    ).to(device)
    s_min = torch.from_numpy(np.asarray(cfg.data.bounding_box_surface.min)).to(
        torch.float32
    ).to(device)

    nx, ny, nz = cfg.model.interp_res
    surf_grid = create_grid(
        s_max, s_min, torch.from_numpy(np.asarray([nx, ny, nz])).to(device)
    )

    normed_stl_vertices_cp = normalize(stl_vertices_t, s_max, s_min)
    surf_grid_normed = normalize(surf_grid, s_max, s_min)

    sdf_surf_grid, _ = signed_distance_field(
        normed_stl_vertices_cp,
        mesh_indices_flattened_t,
        surf_grid_normed,
        use_sign_winding_number=True,
    )
    surf_grid_max_min = torch.stack([s_min, s_max])

    global_params_reference: dict[str, Any] = {
        name: cfg.variables.global_parameters[name]["reference"]
        for name in cfg.variables.global_parameters.keys()
    }
    global_params_types = {
        name: cfg.variables.global_parameters[name]["type"]
        for name in cfg.variables.global_parameters.keys()
    }
    stream_velocity = global_params_reference["inlet_velocity"][0]
    air_density = global_params_reference["air_density"]

    global_params_reference_list: list[float] = []
    for name, typ in global_params_types.items():
        if typ == "vector":
            global_params_reference_list.extend(global_params_reference[name])
        elif typ == "scalar":
            global_params_reference_list.append(global_params_reference[name])
        else:
            raise ValueError(f"Global parameter {name} type not supported")
    global_params_reference_t = torch.from_numpy(
        np.array(global_params_reference_list, dtype=np.float32)
    ).to(device)

    global_params_values_list: list[float] = []
    for key in global_params_types.keys():
        if key == "inlet_velocity":
            global_params_values_list.append(stream_velocity)
        elif key == "air_density":
            global_params_values_list.append(air_density)
        else:
            raise ValueError(f"Global parameter {key} not supported for this recipe")
    global_params_values_t = torch.from_numpy(
        np.array(global_params_values_list, dtype=np.float32)
    ).to(device)

    # Evaluate DoMINO at VTU vertex coordinates so produced predictions line up with point-
    # located GT used for L2 metrics. DoMINO is coordinate-conditioned: feeding ``mesh.points``
    # → predictions sized to ``mesh.n_points`` (no post-hoc interpolation needed downstream).
    if reference_mesh is not None:
        volume_mesh_pv = reference_mesh
    else:
        volume_mesh_pv = pv.read(vtu_path)
    if hasattr(volume_mesh_pv, "cast_to_unstructured_grid") and not isinstance(
        volume_mesh_pv, pv.UnstructuredGrid
    ):
        volume_mesh_pv = volume_mesh_pv.cast_to_unstructured_grid()
    volume_coordinates_np = np.asarray(volume_mesh_pv.points, dtype=np.float32)
    # Drop our local handle so the only owner is the caller-managed ``reference_geometry`` (if any);
    # otherwise the pv.read result becomes unreachable and gets GC'd before the heavy SDF / grid work.
    del volume_mesh_pv
    # ``volume_fields`` is only used in ``domino_volume_test_step`` to allocate ``prediction_vol``
    # via ``zeros_like``; zero placeholder shaped (n_points, n_features) with ``n_points`` equal to
    # ``volume_coordinates_np.shape[0]`` (VTU vertices) and ``n_features`` from ``variables.volume.solution``.
    n_features = sum(
        3 if cfg.variables.volume.solution[name] == "vector" else 1
        for name in volume_variable_names
    )
    volume_fields_np = np.zeros(
        (volume_coordinates_np.shape[0], n_features), dtype=np.float32
    )
    volume_coordinates = (
        torch.from_numpy(volume_coordinates_np).to(torch.float32).to(device)
    )
    volume_fields = torch.from_numpy(volume_fields_np).to(torch.float32).to(device)

    c_max = torch.from_numpy(np.asarray(cfg.data.bounding_box.max)).to(
        torch.float32
    ).to(device)
    c_min = torch.from_numpy(np.asarray(cfg.data.bounding_box.min)).to(
        torch.float32
    ).to(device)

    grid = create_grid(
        c_max, c_min, torch.from_numpy(np.asarray([nx, ny, nz])).to(device)
    )

    if cfg.data.normalize_coordinates:
        volume_coordinates = normalize(volume_coordinates, c_max, c_min)
        grid = normalize(grid, c_max, c_min)
        center_of_mass_normalized = normalize(center_of_mass, c_max, c_min)
        normed_stl_vertices_vol = normalize(stl_vertices_t, c_max, c_min)
    else:
        center_of_mass_normalized = center_of_mass
        normed_stl_vertices_vol = stl_vertices_t

    sdf_grid, _ = signed_distance_field(
        normed_stl_vertices_vol,
        mesh_indices_flattened_t,
        grid,
        use_sign_winding_number=True,
    )

    sdf_nodes, sdf_node_closest_point = signed_distance_field(
        normed_stl_vertices_vol,
        mesh_indices_flattened_t,
        volume_coordinates,
        use_sign_winding_number=True,
    )
    sdf_nodes = sdf_nodes.reshape(-1, 1)
    vol_grid_max_min = torch.stack([c_min, c_max])

    pos_volume_closest = volume_coordinates - sdf_node_closest_point
    pos_volume_center_of_mass = volume_coordinates - center_of_mass_normalized

    geom_centers = stl_vertices_t

    data_dict: dict[str, torch.Tensor] = {
        "pos_volume_closest": pos_volume_closest,
        "pos_volume_center_of_mass": pos_volume_center_of_mass,
        "geometry_coordinates": geom_centers,
        "grid": grid,
        "surf_grid": surf_grid,
        "sdf_grid": sdf_grid,
        "sdf_surf_grid": sdf_surf_grid,
        "sdf_nodes": sdf_nodes,
        "volume_fields": volume_fields,
        "volume_mesh_centers": volume_coordinates,
        "volume_min_max": vol_grid_max_min,
        "surface_min_max": surf_grid_max_min,
        "length_scale": length_scale,
        "global_params_values": torch.unsqueeze(global_params_values_t, -1),
        "global_params_reference": torch.unsqueeze(global_params_reference_t, -1),
    }
    return {k: torch.unsqueeze(v, 0) for k, v in data_dict.items()}


def domino_surface_test_step(
    data_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
    cfg: DictConfig,
    surf_factors: torch.Tensor,
    device: torch.device,
    point_batch_size: int,
) -> torch.Tensor:
    """Surface branch of ``test_step`` in domino ``test.py``."""
    global_params_values = data_dict["global_params_values"]
    global_params_reference = data_dict["global_params_reference"]
    stream_velocity = global_params_reference[:, 0, :]
    air_density = global_params_reference[:, 1, :]
    geo_centers = data_dict["geometry_coordinates"]
    s_grid = data_dict["surf_grid"]
    sdf_surf_grid = data_dict["sdf_surf_grid"]
    surf_max = data_dict["surface_min_max"][:, 1]
    surf_min = data_dict["surface_min_max"][:, 0]

    geo_centers_surf = 2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
    encoding_g_surf = model.geo_rep_surface(geo_centers_surf, s_grid, sdf_surf_grid)

    surface_mesh_centers = data_dict["surface_mesh_centers"]
    surface_normals = data_dict["surface_normals"]
    surface_areas = data_dict["surface_areas"]
    surface_mesh_neighbors = data_dict["surface_mesh_neighbors"]
    surface_neighbors_normals = data_dict["surface_neighbors_normals"]
    surface_neighbors_areas = data_dict["surface_neighbors_areas"]
    pos_surface_center_of_mass = data_dict["pos_surface_center_of_mass"]

    surface_areas = torch.unsqueeze(surface_areas, -1)
    surface_neighbors_areas = torch.unsqueeze(surface_neighbors_areas, -1)

    num_points = surface_mesh_centers.shape[1]
    target_surf = data_dict["surface_fields"]
    prediction_surf = torch.zeros_like(target_surf)

    for p in range(math.ceil(num_points / point_batch_size)):
        start_idx = p * point_batch_size
        end_idx = min((p + 1) * point_batch_size, num_points)
        surface_mesh_centers_batch = surface_mesh_centers[:, start_idx:end_idx]
        surface_mesh_neighbors_batch = surface_mesh_neighbors[:, start_idx:end_idx]
        surface_normals_batch = surface_normals[:, start_idx:end_idx]
        surface_neighbors_normals_batch = surface_neighbors_normals[:, start_idx:end_idx]
        surface_areas_batch = surface_areas[:, start_idx:end_idx]
        surface_neighbors_areas_batch = surface_neighbors_areas[:, start_idx:end_idx]
        pos_surface_center_of_mass_batch = pos_surface_center_of_mass[
            :, start_idx:end_idx
        ]
        geo_encoding_local = model.surface_local_geo_encodings(
            0.5 * encoding_g_surf,
            surface_mesh_centers_batch,
            s_grid,
        )
        pos_encoding = model.fc_p_surf(pos_surface_center_of_mass_batch)
        tpredictions_batch = model.solution_calculator_surf(
            surface_mesh_centers_batch,
            geo_encoding_local,
            pos_encoding,
            surface_mesh_neighbors_batch,
            surface_normals_batch,
            surface_neighbors_normals_batch,
            surface_areas_batch,
            surface_neighbors_areas_batch,
            global_params_values,
            global_params_reference,
        )
        prediction_surf[:, start_idx:end_idx] = tpredictions_batch

    if cfg.model.normalization == "min_max_scaling":
        prediction_surf = unnormalize(prediction_surf, surf_factors[0], surf_factors[1])
    elif cfg.model.normalization == "mean_std_scaling":
        prediction_surf = unstandardize(
            prediction_surf, surf_factors[0], surf_factors[1]
        )
    prediction_surf = (
        prediction_surf * stream_velocity[0, 0] ** 2.0 * air_density[0, 0]
    )
    return prediction_surf


def domino_volume_test_step(
    data_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
    cfg: DictConfig,
    vol_factors: torch.Tensor,
    _device: torch.device,
    point_batch_size: int,
) -> torch.Tensor:
    """Volume branch of ``test_step`` in domino ``test.py`` (volume-only models).

    After normalization undo, converts to physical units using ``variables.volume.solution`` field order:
    velocity-like vectors × ``stream_velocity``, pressure scalar × ``stream_velocity² ρ``,
    nut-like scalar × ``stream_velocity · length_scale`` (matching domino ``test.py`` heuristics per field name).

    Vector fields that are not classified as velocity (name must start with ``u`` or contain ``velocity``)
    raise ``ValueError`` — there is no defined physical scaling for arbitrary extra vector channels.
    """
    length_scale = data_dict["length_scale"]
    global_params_values = data_dict["global_params_values"]
    global_params_reference = data_dict["global_params_reference"]
    stream_velocity = global_params_reference[:, 0, :]
    air_density = global_params_reference[:, 1, :]
    geo_centers = data_dict["geometry_coordinates"]

    p_grid = data_dict["grid"]
    sdf_grid = data_dict["sdf_grid"]
    if "volume_min_max" in data_dict:
        vol_max = data_dict["volume_min_max"][:, 1]
        vol_min = data_dict["volume_min_max"][:, 0]
        geo_centers_vol = 2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1
    else:
        geo_centers_vol = geo_centers

    encoding_g_vol = model.geo_rep_volume(geo_centers_vol, p_grid, sdf_grid)

    volume_mesh_centers = data_dict["volume_mesh_centers"]
    target_vol = data_dict["volume_fields"]
    sdf_nodes = data_dict["sdf_nodes"]
    pos_volume_closest = data_dict["pos_volume_closest"]
    pos_volume_center_of_mass = data_dict["pos_volume_center_of_mass"]

    sdf_scaling_factor = cfg.model.geometry_rep.geo_processor.volume_sdf_scaling_factor

    prediction_vol = torch.zeros_like(target_vol)
    num_points = volume_mesh_centers.shape[1]

    for si in range(math.ceil(num_points / point_batch_size)):
        start_idx = si * point_batch_size
        end_idx = min((si + 1) * point_batch_size, num_points)
        volume_mesh_centers_batch = volume_mesh_centers[:, start_idx:end_idx]
        sdf_nodes_batch = sdf_nodes[:, start_idx:end_idx]
        scaled_sdf_nodes_batch = []
        for j in range(len(sdf_scaling_factor)):
            scaled_sdf_nodes_batch.append(
                scale_sdf(sdf_nodes_batch, sdf_scaling_factor[j])
            )
        scaled_sdf_nodes_batch = torch.cat(scaled_sdf_nodes_batch, dim=-1)

        pos_volume_closest_batch = pos_volume_closest[:, start_idx:end_idx]
        pos_normals_com_batch = pos_volume_center_of_mass[:, start_idx:end_idx]
        geo_encoding_local = model.volume_local_geo_encodings(
            0.5 * encoding_g_vol,
            volume_mesh_centers_batch,
            p_grid,
        )
        if cfg.model.use_sdf_in_basis_func:
            pos_encoding_all = torch.cat(
                (
                    sdf_nodes_batch,
                    scaled_sdf_nodes_batch,
                    pos_volume_closest_batch,
                    pos_normals_com_batch,
                ),
                axis=-1,
            )
        else:
            pos_encoding_all = pos_normals_com_batch

        pos_encoding = model.fc_p_vol(pos_encoding_all)
        tpredictions_batch = model.solution_calculator_vol(
            volume_mesh_centers_batch,
            geo_encoding_local,
            pos_encoding,
            global_params_values,
            global_params_reference,
        )
        prediction_vol[:, start_idx:end_idx] = tpredictions_batch

    if cfg.model.normalization == "min_max_scaling":
        prediction_vol = unnormalize(prediction_vol, vol_factors[0], vol_factors[1])
    elif cfg.model.normalization == "mean_std_scaling":
        prediction_vol = unstandardize(prediction_vol, vol_factors[0], vol_factors[1])

    # Physical scaling (domino ``test.py``): match ``variables.volume.solution`` order, not fixed indices.
    sv = stream_velocity[0, 0]
    rho = air_density[0, 0]
    ell = length_scale[0]
    offset_phys = 0
    for name, typ in cfg.variables.volume.solution.items():
        kind = _volume_solution_field_kind(name, typ)
        if typ == "vector":
            sl = slice(offset_phys, offset_phys + 3)
            if kind == "velocity_vector":
                prediction_vol[:, :, sl] *= sv
            elif kind == "other":
                raise ValueError(
                    "volume physical scaling: vector field "
                    f"{name!r} is not classified as velocity (name should start with 'u' or "
                    "contain 'velocity'). Extra vector outputs have no defined physical scaling "
                    "— rename to a velocity-like key or remove the field from ``variables.volume.solution``."
                )
            offset_phys += 3
        else:
            if kind == "pressure_scalar":
                prediction_vol[:, :, offset_phys] *= sv**2.0 * rho
            elif kind == "nut_scalar":
                prediction_vol[:, :, offset_phys] *= sv * ell
            offset_phys += 1

    if offset_phys != prediction_vol.shape[-1]:
        raise ValueError(
            "volume physical scaling walk does not span model output channels; "
            f"combined width from ``variables.volume.solution`` is {offset_phys}, "
            f"tensor last dim is {prediction_vol.shape[-1]}."
        )

    return prediction_vol


def domino_volume_predictions_to_canonical(
    pred: torch.Tensor, cfg: DictConfig
) -> Dict[str, Any]:
    """Map DoMINO volume output (physical units) to canonical keys using ``variables.volume.solution`` order.

    Assumes the usual DrivAer layout (velocity vector, pressure scalar, nut scalar) when names match.
    Non-velocity vector entries in ``variables.volume.solution`` raise ``ValueError`` (same rule as
    :func:`domino_volume_test_step`). Scalar fields classified as ``other`` are still passed through
    under their config key name in ``extra``.
    """
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    arr = pred.cpu().numpy().astype(np.float32)

    canonical_kw: dict[str, np.ndarray] = {}
    canonical_first_field: dict[str, str] = {}
    extra: dict[str, np.ndarray] = {}

    def _put_canonical(key: str, value: np.ndarray, field_name: str) -> None:
        if key in canonical_kw:
            raise ValueError(
                "two ``variables.volume.solution`` fields map to the same canonical output "
                f"{key!r}: {canonical_first_field[key]!r} and {field_name!r}. "
                "Rename fields so substring heuristics classify at most one field per canonical quantity."
            )
        canonical_first_field[key] = field_name
        canonical_kw[key] = value

    offset = 0
    for name, typ in cfg.variables.volume.solution.items():
        kind = _volume_solution_field_kind(name, typ)
        if typ == "vector":
            chunk = arr[:, offset : offset + 3]
            offset += 3
            if kind == "velocity_vector":
                _put_canonical("velocity", chunk, name)
            else:
                raise ValueError(
                    "volume canonical mapping: vector field "
                    f"{name!r} is not classified as velocity (name should start with 'u' or "
                    "contain 'velocity'). Extra vector outputs have no defined physical scaling in "
                    "the DoMINO volume path — rename or remove the field from ``variables.volume.solution``."
                )
        else:
            chunk = arr[:, offset]
            offset += 1
            if kind == "pressure_scalar":
                _put_canonical("pressure", chunk, name)
            elif kind == "nut_scalar":
                _put_canonical("turbulent_viscosity", chunk, name)
            else:
                extra[name] = chunk

    expected = arr.shape[-1]
    if offset != expected:
        raise ValueError(
            f"volume solution channels ({expected}) do not match ``variables.volume.solution`` "
            f"layout (combined width {offset}); check field names/order against model output "
            f"so canonical maps do not overlap or omit slices."
        )

    return build_predictions_dict(**canonical_kw, **extra)
