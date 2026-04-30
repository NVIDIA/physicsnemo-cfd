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

from physicsnemo.cfd.postprocessing_tools.metrics.l2_errors import triangulate_surface_mesh


def read_stl_geometry(stl_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Read STL and return stl_coordinates, stl_faces, stl_centers for SDF/center of mass."""
    mesh_raw = pv.read(stl_path)
    mesh = triangulate_surface_mesh(mesh_raw)
    stl_coordinates = torch.from_numpy(np.asarray(mesh.points)).to(
        device=device, dtype=torch.float32
    )
    faces = np.asarray(mesh.regular_faces, dtype=np.int64)
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
    vtp_path: str,
    device: torch.device,
    n_output_fields: int = 4,
    *,
    mesh: pv.DataSet | None = None,
) -> dict[str, torch.Tensor]:
    """Read VTP surface: cell centers, normals, areas, dummy surface_fields.

    When VTK cell normals are absent (``.cell_normals is None``), computes them via
    :meth:`~pyvista.PolyData.compute_normals` with ``cell_normals=True``,
    ``point_normals=False`` — same idea as XMGN when point ``Normals`` are missing.

    Parameters
    ----------
    mesh
        If set (e.g. from :attr:`~physicsnemo.cfd.evaluation.datasets.schema.CanonicalCase.reference_geometry`),
        skips ``pv.read(vtp_path)`` and uses this dataset (converted to ``PolyData`` when needed).
    """
    if mesh is None:
        mesh = pv.read(vtp_path)
    elif not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if mesh.cell_normals is None:
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False)
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
    vtu_path: str,
    device: torch.device,
    n_output_fields: int = 5,
    *,
    mesh: pv.DataSet | None = None,
) -> dict[str, torch.Tensor]:
    """Read VTU volume mesh: cell centers and dummy ``volume_fields`` (inference_on_vtk layout).

    Parameters
    ----------
    mesh
        Optional in-memory mesh to avoid ``pv.read(vtu_path)`` (e.g. ``CanonicalCase.reference_geometry``).
    """
    if mesh is None:
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


def _find_stl_in_dir(run_dir: Path, run_idx: int) -> Path:
    """Resolve one STL geometry in *run_dir* for this run index.

    Shared by :func:`build_surface_data_dict` and :func:`build_volume_data_dict` so the
    same directory with multiple naming conventions yields the **same** STL for surface
    and volume. Explicit ``drivaer_*`` names first; then sorted globs for determinism.

    Order: ``drivaer_{run_idx}_single_solid.stl`` → ``drivaer_{run_idx}.stl`` → first
    sorted ``*_single_solid.stl`` → first sorted ``*.stl``.
    """
    run_dir = Path(run_dir)
    for stem in (
        f"drivaer_{run_idx}_single_solid",
        f"drivaer_{run_idx}",
    ):
        p = run_dir / f"{stem}.stl"
        if p.is_file():
            return p
    for pattern in ("*_single_solid.stl", "*.stl"):
        for candidate in sorted(run_dir.glob(pattern)):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No STL file found in {run_dir} for run_idx {run_idx}")


def build_volume_data_dict(
    run_dir: Path,
    vtu_path: str,
    device: torch.device,
    air_density: float,
    stream_velocity: float,
    run_idx: int = 1,
    n_output_fields: int = 5,
    *,
    reference_mesh: pv.DataSet | None = None,
) -> dict[str, torch.Tensor]:
    """Build data dict for volume inference: STL + VTU + flow params (DrivAer-style run dir).

    STL resolution matches :func:`build_surface_data_dict` — ``drivaer_<run_idx>.stl``,
    ``drivaer_<run_idx>_single_solid.stl``, then ``*_single_solid.stl``, then any ``*.stl``.
    """
    stl_path = _find_stl_in_dir(run_dir, run_idx)
    data_dict = read_stl_geometry(str(stl_path), device)
    data_dict.update(
        read_volume_from_vtu(
            vtu_path, device, n_output_fields=n_output_fields, mesh=reference_mesh
        )
    )
    data_dict["air_density"] = torch.tensor([air_density], device=device, dtype=torch.float32)
    data_dict["stream_velocity"] = torch.tensor(
        [stream_velocity], device=device, dtype=torch.float32
    )
    return data_dict


def build_surface_data_dict(
    run_dir: Path,
    vtp_path: str,
    device: torch.device,
    air_density: float,
    stream_velocity: float,
    run_idx: int = 1,
    *,
    reference_mesh: pv.DataSet | None = None,
) -> dict[str, torch.Tensor]:
    """Build data dict for surface inference: STL + VTP + flow params. Finds STL in run_dir."""
    stl_path = _find_stl_in_dir(run_dir, run_idx)
    data_dict = read_stl_geometry(str(stl_path), device)
    data_dict.update(read_surface_from_vtp(vtp_path, device, mesh=reference_mesh))
    data_dict["air_density"] = torch.tensor([air_density], device=device, dtype=torch.float32)
    data_dict["stream_velocity"] = torch.tensor(
        [stream_velocity], device=device, dtype=torch.float32
    )
    return data_dict


def run_id_from_case_id(case_id: str) -> int:
    """Parse run index from *case_id* (``run_<n>`` or decimal *n*, e.g. ``'run_3'``, ``'3'``).

    Raises
    ------
    ValueError
        If *case_id* is empty, not ``run_<integer>``, and not a decimal integer string.
        Used to choose ``drivaer_<n>*.stl`` in the case directory; invalid IDs must not
        fall back to run index 1.
    """
    s = str(case_id).strip()
    if not s:
        raise ValueError("case_id is empty")

    if s.startswith("run_"):
        suffix = s[4:]
        if not suffix:
            raise ValueError(f"invalid run_* case_id (missing index): {case_id!r}")
        try:
            return int(suffix)
        except ValueError as e:
            raise ValueError(
                f"invalid run_* case_id (expected run_<integer>, got {case_id!r})"
            ) from e

    try:
        return int(s)
    except ValueError as e:
        raise ValueError(
            f"case_id must be 'run_<n>' or a decimal integer (got {case_id!r})"
        ) from e
