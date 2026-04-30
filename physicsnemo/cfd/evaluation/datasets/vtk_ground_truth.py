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

"""Load reference fields from VTK: surface VTP (pressure / WSS) and volume VTU (pressure / velocity / νₜ)."""

from typing import Literal

import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors


# Generic external-aero CFD field names (reference, not ``*Pred``). Adapters with their own
# stronger conventions (e.g. DrivAer ``*MeanTrim``) pass adapter-specific tuples through
# ``extract_*_from_mesh(..., *_names=...)`` rather than relying on these defaults.
DEFAULT_PRESSURE_NAMES = (
    "pMeanTrim",
    "pMean",
    "Pressure",
    "pressure",
    "p",
)
DEFAULT_SHEAR_NAMES = (
    "wallShearStressMeanTrim",
    "WallShearStress",
    "wallShearStress",
    "wallShearStressMean",
)

# Volume RANS / LES-style names (reference CFD). These are dataset-agnostic generic CFD
# conventions; adapters that ship a more specific convention (e.g. DrivAer ``*MeanTrim``)
# pass their own tuples through ``extract_volume_fields_from_mesh(..., velocity_names=...)``.
DEFAULT_TURBULENT_VISCOSITY_NAMES = (
    "nutMean",
    "nut",
    "turbulent_viscosity",
    "TurbulentViscosity",
    "nuTilde",
)
DEFAULT_VOLUME_VELOCITY_NAMES = (
    "UMean",
    "U",
    "velocity",
    "Velocity",
)
# Volume pressure uses the same canonical key ``pressure`` as surface (domain disambiguates).
DEFAULT_VOLUME_PRESSURE_NAMES = DEFAULT_PRESSURE_NAMES


def _find_array(
    data: pv.DataSetAttributes,
    candidates: tuple[str, ...],
) -> str | None:
    for name in candidates:
        if name in data:
            return name
    return None


def _pressure_to_1d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    return a.reshape(-1)


def _shear_to_n3(arr: np.ndarray, n_expected: int) -> np.ndarray | None:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        if a.size == n_expected * 3:
            return a.reshape(n_expected, 3)
        return None
    if a.ndim == 2 and a.shape[1] == 3:
        if a.shape[0] == n_expected:
            return a
    return None


def _extract_fields_at_location(
    mesh: pv.PolyData,
    loc_name: Literal["cell", "point"],
    pressure_names: tuple[str, ...],
    shear_names: tuple[str, ...],
) -> dict[str, np.ndarray] | None:
    """Try to read pressure / shear at one mesh location; return dict or None if nothing valid."""
    data = mesh.cell_data if loc_name == "cell" else mesh.point_data
    n = mesh.n_cells if loc_name == "cell" else mesh.n_points
    if n == 0:
        return None

    pkey = _find_array(data, pressure_names)
    skey = _find_array(data, shear_names)
    out: dict[str, np.ndarray] = {}

    if pkey is not None:
        p = _pressure_to_1d(np.asarray(data[pkey]))
        if p.size == n:
            out["pressure"] = p.astype(np.float32)
    if skey is not None:
        w = _shear_to_n3(np.asarray(data[skey]), n)
        if w is not None:
            out["shear_stress"] = w.astype(np.float32)

    return out if out else None


def resample_cell_ground_truth_to_points(
    mesh: pv.PolyData,
    ground_truth_cell: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map cell-centered GT arrays to mesh points using VTK cell→point averaging.

    Use when reference fields exist only on cells but predictions are point-based
    (e.g. MeshGraphNet) so L2 metrics share the same dof count as ``mesh.n_points``.
    """
    n_c = mesh.n_cells
    m = mesh.copy(deep=True)
    # Temporary names to avoid clobbering existing arrays
    if "pressure" in ground_truth_cell:
        p = np.asarray(ground_truth_cell["pressure"]).ravel()
        if p.size != n_c:
            raise ValueError(
                f"pressure length {p.size} != n_cells {n_c} for cell→point resampling"
            )
        m.cell_data["_cfd_gt_pressure"] = p.astype(np.float32)
    if "shear_stress" in ground_truth_cell:
        w = np.asarray(ground_truth_cell["shear_stress"])
        if w.shape[0] != n_c or w.shape[-1] != 3:
            raise ValueError(
                f"shear_stress shape {w.shape} incompatible with n_cells={n_c}"
            )
        m.cell_data["_cfd_gt_shear"] = w.astype(np.float32)

    try:
        m_pt = m.cell_data_to_point_data(pass_cell_data=True)
    except Exception:
        m_pt = m.cell_data_to_point_data()

    out: dict[str, np.ndarray] = {}
    if "_cfd_gt_pressure" in m_pt.point_data:
        out["pressure"] = np.asarray(m_pt.point_data["_cfd_gt_pressure"], dtype=np.float32).ravel()
    if "_cfd_gt_shear" in m_pt.point_data:
        out["shear_stress"] = np.asarray(m_pt.point_data["_cfd_gt_shear"], dtype=np.float32)

    if set(out.keys()) == set(ground_truth_cell.keys()):
        return out

    # VTK filter sometimes omits custom arrays; map cell values to points by nearest cell center.
    centers = mesh.cell_centers().points
    pts = np.asarray(mesh.points, dtype=np.float64)
    nbr = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(centers)
    _, cell_idx = nbr.kneighbors(pts)
    ci = cell_idx.ravel().astype(np.int64)
    out_nn: dict[str, np.ndarray] = {}
    if "pressure" in ground_truth_cell:
        p = np.asarray(ground_truth_cell["pressure"], dtype=np.float64).ravel()
        if p.size == n_c:
            out_nn["pressure"] = p[ci].astype(np.float32)
    if "shear_stress" in ground_truth_cell:
        w = np.asarray(ground_truth_cell["shear_stress"], dtype=np.float64)
        if w.shape[0] == n_c and w.shape[-1] == 3:
            out_nn["shear_stress"] = w[ci].astype(np.float32)
    return out_nn


def extract_pressure_wss_from_mesh(
    mesh: pv.PolyData,
    data_type: Literal["auto", "cell", "point"] = "auto",
    pressure_names: tuple[str, ...] = DEFAULT_PRESSURE_NAMES,
    shear_names: tuple[str, ...] = DEFAULT_SHEAR_NAMES,
) -> tuple[dict[str, np.ndarray] | None, Literal["cell", "point"] | None]:
    """Read reference pressure and shear stress from mesh cell_data / point_data.

    ``data_type``:

    - ``auto``: tries **cell** first, then **point** (legacy behavior).
    - ``cell``: cell first, then point (same as today for two-pass).
    - ``point``: **point** first. If only cell reference fields exist, they are resampled to
      points (``cell_data_to_point_data``) so array lengths match ``mesh.n_points`` — required
      for point-based models vs metrics.

    Returns:
        (ground_truth_dict_or_none, location_or_none). ``location`` is ``"point"`` or
        ``"cell"`` describing where the returned arrays live (after any resampling).
    """
    if data_type == "point":
        pt = _extract_fields_at_location(mesh, "point", pressure_names, shear_names)
        if pt:
            return (pt, "point")
        cell_gt = _extract_fields_at_location(mesh, "cell", pressure_names, shear_names)
        if cell_gt:
            try:
                pt_resampled = resample_cell_ground_truth_to_points(mesh, cell_gt)
            except Exception:
                pt_resampled = {}
            if pt_resampled:
                return (pt_resampled, "point")
        return (None, None)

    if data_type == "cell":
        order: list[Literal["cell", "point"]] = ["cell", "point"]
    else:
        order = ["cell", "point"]

    for loc_name in order:
        out = _extract_fields_at_location(
            mesh, loc_name, pressure_names, shear_names
        )
        if out:
            return (out, loc_name)

    return (None, None)


def _scalar_field_to_1d(arr: np.ndarray, n: int) -> np.ndarray | None:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    a = a.reshape(-1)
    if a.size != n:
        return None
    return a.astype(np.float32)


def _volume_vec3(arr: np.ndarray, n: int) -> np.ndarray | None:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1 and a.size == n * 3:
        return a.reshape(n, 3).astype(np.float32)
    if a.ndim == 2 and a.shape == (n, 3):
        return a.astype(np.float32)
    return None


def _extract_volume_at_location(
    mesh: pv.DataSet,
    loc_name: Literal["cell", "point"],
    turbulent_viscosity_names: tuple[str, ...],
    velocity_names: tuple[str, ...],
    pressure_names: tuple[str, ...],
) -> dict[str, np.ndarray] | None:
    data = mesh.cell_data if loc_name == "cell" else mesh.point_data
    n = mesh.n_cells if loc_name == "cell" else mesh.n_points
    if n == 0:
        return None
    out: dict[str, np.ndarray] = {}
    pk = _find_array(data, pressure_names)
    if pk is not None:
        p = _pressure_to_1d(np.asarray(data[pk]))
        if p.size == n:
            out["pressure"] = p.astype(np.float32)
    nk = _find_array(data, turbulent_viscosity_names)
    if nk is not None:
        nut = _scalar_field_to_1d(np.asarray(data[nk]), n)
        if nut is not None:
            out["turbulent_viscosity"] = nut
    vk = _find_array(data, velocity_names)
    if vk is not None:
        vel = _volume_vec3(np.asarray(data[vk]), n)
        if vel is not None:
            out["velocity"] = vel
    return out if out else None


def extract_volume_fields_from_mesh(
    mesh: pv.DataSet,
    data_type: Literal["auto", "cell", "point"] = "auto",
    turbulent_viscosity_names: tuple[str, ...] | None = None,
    velocity_names: tuple[str, ...] | None = None,
    pressure_names: tuple[str, ...] | None = None,
) -> tuple[dict[str, np.ndarray] | None, Literal["cell", "point"] | None]:
    """Read reference fields from volume ``cell_data`` / ``point_data``.

    Scalar pressure is stored under the canonical key ``pressure`` (same as surface;
    the inference domain disambiguates surface vs volume).

    Uses defaults when a name tuple is ``None``. Pass ``()`` to skip reading that field group.
    """
    p_names = (
        DEFAULT_VOLUME_PRESSURE_NAMES if pressure_names is None else pressure_names
    )
    nut_names = (
        DEFAULT_TURBULENT_VISCOSITY_NAMES
        if turbulent_viscosity_names is None
        else turbulent_viscosity_names
    )
    vel_names = DEFAULT_VOLUME_VELOCITY_NAMES if velocity_names is None else velocity_names

    if data_type == "point":
        order: list[Literal["cell", "point"]] = ["point", "cell"]
    elif data_type == "cell":
        order = ["cell", "point"]
    else:
        order = ["cell", "point"]

    for loc_name in order:
        got = _extract_volume_at_location(mesh, loc_name, nut_names, vel_names, p_names)
        if got:
            return (got, loc_name)
    return (None, None)
