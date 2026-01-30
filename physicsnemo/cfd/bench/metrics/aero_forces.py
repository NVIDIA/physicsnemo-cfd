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

import numpy as np
import pyvista as pv


def compute_force_coefficients(
    normals: np.ndarray,
    area: np.ndarray,
    coeff: float,
    p: np.ndarray,
    wss: np.ndarray,
    force_direction: np.ndarray = np.array([1, 0, 0]),
):
    """
    Computes force coefficients for a given mesh. Output includes the pressure and skin
    friction components. Can be used to compute lift and drag.
    For drag, use the `force_direction` as the direction of the motion,
    e.g. [1, 0, 0] for flow in x direction.
    For lift, use the `force_direction` as the direction perpendicular to the motion,
    e.g. [0, 1, 0] for flow in x direction and weight in y direction.

    Parameters:
    -----------
    normals: np.ndarray
        The surface normals on cells of the mesh
    area: np.ndarray
        The surface areas of each cell
    coeff: float
        Reciprocal of dynamic pressure times the frontal area, i.e. 2/(A * rho * U^2)
    p: np.ndarray
        Pressure distribution on the mesh (on each cell)
    wss: np.ndarray
        Wall shear stress distribution on the mesh (on each cell)
    force_direction: np.ndarray
        Direction to compute the force, default is np.array([1, 0, 0])

    Returns:
    --------
    c_total: float
        Computed total force coefficient
    c_p: float
        Computed pressure force coefficient
    c_f: float
        Computed skin friction coefficient
    """

    # Compute coefficients
    c_p = coeff * np.sum(np.dot(normals, force_direction) * area * p)
    c_f = -coeff * np.sum(np.dot(wss, force_direction) * area)

    # Compute total force coefficients
    c_total = c_p + c_f

    return c_total, c_p, c_f


def compute_drag_and_lift(
    mesh,
    pressure_field="p",
    wss_field="wss",
    coeff=1.0,
    drag_direction=[1, 0, 0],
    lift_direction=[0, 0, 1],
    dtype="cell",
):
    """Compute Drag and Lift for a given mesh

    Parameters
    ----------
    mesh :
        PyVista Dataset
    pressure_field : str, optional
        Name of the pressure field in the dataset, by default "p"
    wss_field : str, optional
        Name of the wall shear stress field in the dataset, by default "wss"
    coeff : float, optional
        Coefficient to multiply the forces. Typically 2 / (rho * u * u * ref_area), by default 1.0
    drag_direction : list, optional
        Direction for drag calculation, by default [1, 0, 0]
    lift_direction : list, optional
        Direction for lift calculation, by default [0, 0, 1]
    dtype : str, optional
        Wether to compute drag from cell data or point data, by default "cell"

    Returns
    -------
    _type_
        Drag and Lift coefficients and their pressure and skin friction components

    """
    if dtype == "cell":
        mesh = mesh.compute_normals()
        mesh = mesh.compute_cell_sizes()

        p = mesh.cell_data[pressure_field]
        wss = mesh.cell_data[wss_field]

        normals = mesh["Normals"]
        areas = mesh["Area"]

        cd, cd_p, cd_f = compute_force_coefficients(
            normals, areas, coeff, p, wss, np.array(drag_direction)
        )
        cl, cl_p, cl_f = compute_force_coefficients(
            normals, areas, coeff, p, wss, np.array(lift_direction)
        )
    elif dtype == "point":
        # Assuming the point clouds are written using Modulus Sym
        normals = -1 * np.stack(
            [
                mesh.point_data["normal_x"],
                mesh.point_data["normal_y"],
                mesh.point_data["normal_z"],
            ],
            axis=1,
        )
        areas = mesh.point_data["area"]

        p = mesh.point_data[pressure_field]
        wss = mesh.point_data[wss_field]

        cd, cd_p, cd_f = compute_force_coefficients(
            normals, areas, coeff, p, wss, np.array(drag_direction)
        )
        cl, cl_p, cl_f = compute_force_coefficients(
            normals, areas, coeff, p, wss, np.array(lift_direction)
        )
    else:
        raise ValueError("dtype must be either cell or point")

    return cd, cd_p, cd_f, cl, cl_p, cl_f
