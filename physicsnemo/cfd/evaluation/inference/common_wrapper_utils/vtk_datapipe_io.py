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

"""VTK/STL → tensor dicts for CAE inference datapipes (surface and volume).

Builds the same keys expected by ``TransolverDataPipe.process_data`` as in
``examples/cfd/external_aerodynamics/transformer_models/src/inference_on_vtk.py``.
Surface: STL + boundary VTP. Volume: STL + volume VTU. DrivAer-style ``run_*`` layout;
used by wrappers that feed those tensors into a datapipes-based model (e.g. Transolver,
GeoTransolver).
"""

from pathlib import Path

import numpy as np
import pyvista as pv
import torch


def read_stl_geometry(stl_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Read STL and return stl_coordinates, stl_faces, stl_centers for SDF/center of mass."""
    mesh = pv.read(stl_path)
    stl_coordinates = torch.from_numpy(np.asarray(mesh.points)).to(
        device=device, dtype=torch.float32
    )
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    stl_faces = torch.from_numpy(faces.flatten()).to(device=device, dtype=torch.int32)
    stl_centers = torch.from_numpy(np.asarray(mesh.cell_centers().points)).to(
        device=device, dtype=torch.float32
    )
    return {
        "stl_coordinates": stl_coordinates,
        "stl_faces": stl_faces,
        "stl_centers": stl_centers,
    }


def read_surface_from_vtp(
    vtp_path: str, device: torch.device, n_output_fields: int = 4
) -> dict[str, torch.Tensor]:
    """Read VTP surface: cell centers, normals, areas, dummy surface_fields."""
    mesh = pv.read(vtp_path)
    surface_mesh_centers = torch.from_numpy(np.asarray(mesh.cell_centers().points)).to(
        device=device, dtype=torch.float32
    )
    normals = np.asarray(mesh.cell_normals)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    surface_normals = torch.from_numpy(normals).to(device=device, dtype=torch.float32)
    cell_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    surface_areas = torch.from_numpy(np.asarray(cell_sizes.cell_data["Area"])).to(
        device=device, dtype=torch.float32
    )
    num_cells = surface_mesh_centers.shape[0]
    surface_fields = torch.zeros(
        (num_cells, n_output_fields), device=device, dtype=torch.float32
    )
    return {
        "surface_mesh_centers": surface_mesh_centers,
        "surface_normals": surface_normals,
        "surface_areas": surface_areas,
        "surface_fields": surface_fields,
    }


def read_volume_from_vtu(
    vtu_path: str, device: torch.device, n_output_fields: int = 5
) -> dict[str, torch.Tensor]:
    """Read VTU volume mesh: cell centers and dummy ``volume_fields`` (inference_on_vtk layout)."""
    mesh = pv.read(vtu_path)
    volume_mesh_centers = torch.from_numpy(np.asarray(mesh.cell_centers().points)).to(
        device=device, dtype=torch.float32
    )
    num_cells = volume_mesh_centers.shape[0]
    volume_fields = torch.zeros(
        (num_cells, n_output_fields), device=device, dtype=torch.float32
    )
    return {
        "volume_mesh_centers": volume_mesh_centers,
        "volume_fields": volume_fields,
    }


def build_volume_data_dict(
    run_dir: Path,
    vtu_path: str,
    device: torch.device,
    air_density: float,
    stream_velocity: float,
    run_idx: int = 1,
    n_output_fields: int = 5,
) -> dict[str, torch.Tensor]:
    """Build data dict for volume inference: STL + VTU + flow params (DrivAer-style run dir)."""
    stl_path = run_dir / f"drivaer_{run_idx}_single_solid.stl"
    if not stl_path.exists():
        stl_files = list(run_dir.glob("*_single_solid.stl"))
        if not stl_files:
            raise FileNotFoundError(f"No STL file found in {run_dir}")
        stl_path = stl_files[0]
    data_dict = read_stl_geometry(str(stl_path), device)
    data_dict.update(read_volume_from_vtu(vtu_path, device, n_output_fields=n_output_fields))
    data_dict["air_density"] = torch.tensor([air_density], device=device, dtype=torch.float32)
    data_dict["stream_velocity"] = torch.tensor(
        [stream_velocity], device=device, dtype=torch.float32
    )
    return data_dict


def _find_stl_in_dir(run_dir: Path, run_idx: int) -> Path:
    """Find an STL file in *run_dir* using progressively looser name patterns."""
    candidates = [
        run_dir / f"drivaer_{run_idx}.stl",
        run_dir / f"drivaer_{run_idx}_single_solid.stl",
    ]
    for p in candidates:
        if p.exists():
            return p
    for pattern in ("*_single_solid.stl", "*.stl"):
        globs = list(run_dir.glob(pattern))
        if globs:
            return globs[0]
    raise FileNotFoundError(f"No STL file found in {run_dir} for run_idx {run_idx}")


def build_surface_data_dict(
    run_dir: Path,
    vtp_path: str,
    device: torch.device,
    air_density: float,
    stream_velocity: float,
    run_idx: int = 1,
) -> dict[str, torch.Tensor]:
    """Build data dict for surface inference: STL + VTP + flow params. Finds STL in run_dir."""
    stl_path = _find_stl_in_dir(run_dir, run_idx)
    data_dict = read_stl_geometry(str(stl_path), device)
    data_dict.update(read_surface_from_vtp(vtp_path, device))
    data_dict["air_density"] = torch.tensor([air_density], device=device, dtype=torch.float32)
    data_dict["stream_velocity"] = torch.tensor(
        [stream_velocity], device=device, dtype=torch.float32
    )
    return data_dict


def run_id_from_case_id(case_id: str) -> int:
    """Parse run index from case_id (e.g. 'run_1' -> 1)."""
    if case_id.startswith("run_"):
        return int(case_id.split("_")[1])
    try:
        return int(case_id)
    except ValueError:
        return 1
