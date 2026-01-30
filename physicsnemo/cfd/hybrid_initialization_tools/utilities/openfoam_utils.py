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
import re
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import pyvista as pv


def read_openfoam_mesh(run_directory: Path | str) -> pv.UnstructuredGrid:
    """
    Read an OpenFOAM mesh from a run directory.

    Parameters
    ----------
    run_directory : Path or str
        The path to the run directory containing the OpenFOAM case.

    Returns
    -------
    pyvista.UnstructuredGrid
        The OpenFOAM mesh.
    """

    run_directory = Path(run_directory)

    # Create an empty case.foam file to help pyvista locate the OpenFOAM case
    foam_file = run_directory / "case.foam"

    if not foam_file.exists():
        created_foam_file = True
        foam_file.touch()
    else:
        created_foam_file = False

    reader = pv.POpenFOAMReader(foam_file)
    reader.enable_all_patch_arrays()
    reader.set_active_time_value(np.max(reader.time_values))  # Use the last time step
    mesh = reader.read()

    if created_foam_file:  # Clean up the case.foam file, if it was created here.
        foam_file.unlink()

    return mesh["internalMesh"]


def read_openfoam_cell_centers(file_path: Path | str) -> np.ndarray:
    """
    Parses an OpenFOAM-formatted vector field data file.

    Example input: the file `C` produced using the following OpenFOAM command:
    `postProcess -func writeCellCentres -constant`. This will create an array with positions of the cell centers.

    Parameters
    ----------
    file_path : Path or str
        An OpenFOAM-formatted vector field data file.

    Returns
    -------
    np.ndarray
        A Nx3 NumPy float array containing the data in that field.

    Raises
    ------
    ValueError
        If no tuples are found in the specified block.
    """
    inside_block = False
    data: list[tuple[float, float, float]] = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if line == "(":  # Start collecting tuples
                inside_block = True
                continue

            if line == ")" and inside_block:  # End collecting tuples
                inside_block = False
                break

            if inside_block:  # Collect tuples
                try:  # Attempt to parse the line as a tuple
                    # Match the format "(x y z)" and split the line
                    line_data = line.strip("()").split()
                    if len(line_data) == 3:
                        data.append(tuple(map(float, line_data)))
                except ValueError:
                    pass  # Ignore lines that can't be parsed as tuples

    # Convert the collected data to a NumPy array
    if not data:
        raise ValueError("No tuples found in the specified block.")

    return np.array(data, dtype=float)


def replace_internal_field(
    file_path: Path | str, output_path: Path | str, array: np.ndarray
) -> None:
    """
    Replace the internalField block in an OpenFOAM field file with a new
    internalField.

    Parameters
    ----------
    file_path : Path or str
        The path to the OpenFOAM field file to replace the internalField block
        in.
    output_path : Path or str
        The path to the file to write the modified internalField block to. May
        be the same as file_path, if you want to overwrite the original file.
    array : np.ndarray
        The new internalField array. A 1D array of shape (n_cells,) for scalar
        fields, or a 2D array of shape (n_cells, 3) for vector fields.
    """
    if array.ndim == 1:
        is_vector = False
    elif array.ndim == 2 and array.shape[-1] == 3:
        is_vector = True
    else:
        raise NotImplementedError(
            "So far, this is only expected to work with scalar fields or 3D vector fields."
        )

    # Generate the new internalField block
    n_cells = len(array)
    fieldtype = "vector" if is_vector else "scalar"
    new_internal_field_lines: list[str] = [
        f"internalField   nonuniform List<{fieldtype}> \n",
        f"{n_cells}\n",
        f"(\n",
    ]
    if is_vector:
        new_internal_field_lines += [
            f"({x:.12f} {y:.12f} {z:.12f})\n" for x, y, z in array
        ]
    else:
        new_internal_field_lines += [f"{value:.12f}\n" for value in array]
    new_internal_field_lines.append(");\n")

    # Read the file contents
    with open(file_path, "r") as file:
        content = file.readlines()

    # Locate and replace the internalField block
    start_idx, end_idx = None, None
    dynamic_internalField_found = False
    for idx, line in enumerate(content):
        if "internalField" in line and start_idx is None:
            start_idx = idx
        if start_idx is not None and ";" in line and end_idx is None:
            end_idx = idx
        if "$internalField" in line:
            dynamic_internalField_found = True

    if start_idx is None or end_idx is None:
        raise ValueError("internalField block not found in the input file.")

    # Replace any dynamic $internalField tags
    if dynamic_internalField_found:
        if start_idx != end_idx:
            raise NotImplementedError(
                "If you use $internalField dynamic links, the original internalField must be single-line."
            )
        original_internalField = re.sub(
            r"^\s*internalField\s+|\s*;\s*$", "", content[start_idx]
        )
        for idx, line in enumerate(content):
            content[idx] = content[idx].replace(
                "$internalField", original_internalField
            )

    # Replace the block
    content = content[:start_idx] + new_internal_field_lines + content[end_idx + 1 :]

    # Write the modified content to the output file
    with open(output_path, "w") as file:
        file.writelines(content)


def interpolate_fields(
    original_coords: np.ndarray,
    target_coords: np.ndarray,
    fields: dict[str, np.ndarray],
    n_neighbors: int = 16,
    idw_power: int = 4,
) -> dict[str, np.ndarray]:
    """
    Interpolates fields from original_coords to target_coords using inverse distance weighting.

    Parameters
    ----------
    original_coords : np.ndarray
        Array of shape (n_samples, n_dims) representing the coordinates of the original points.
    target_coords : np.ndarray
        Array of shape (m_samples, n_dims) representing the coordinates of the target points.
    fields : dict of str to np.ndarray
        Dictionary where keys are field names and values are arrays of shape (n_samples,) representing the field values at the original points.
    n_neighbors : int, optional
        Number of nearest neighbors to use for interpolation. Default is 16.
    idw_power : int, optional
        Power parameter for inverse distance weighting. Default is 4.

    Returns
    -------
    dict of str to np.ndarray
        Dictionary where keys are field names and values are arrays of shape (m_samples,) representing the interpolated field values at the target points.
    """
    # Find nearest neighbors
    knn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(original_coords)

    # Compute distances and indices of nearest neighbors for target points
    distances, indices = knn.kneighbors(target_coords)

    # Compute weights as inverse distances
    distances += 1e-10  # Avoid division by zero
    relative_distances = distances / np.min(distances, axis=1, keepdims=True)
    weights = (relative_distances) ** (-idw_power)
    weights /= weights.sum(axis=1, keepdims=True)

    # Interpolate each field using precomputed weights
    return {
        k: np.einsum("iw,iw...->i...", weights, v[indices]) for k, v in fields.items()
    }


def parse_initial_conditions(file_path: Path) -> dict:
    """
    Parse OpenFOAM initialConditions file to extract flow parameters.

    Parameters
    ----------
    file_path : Path
        Path to the initialConditions file.

    Returns
    -------
    dict
        Dictionary containing the parsed flow parameters:
        - "p": pressure
        - "U": velocity
        - "k": turbulent kinetic energy
        - "omega": turbulent specific dissipation rate.

    Raises
    ------
    ValueError
        If any of the required values are not found in the initialConditions file.
    """
    with open(file_path, "r") as f:
        content = f.read()
    # Use regex to extract values
    velocity_match = re.search(r"flowVelocity\s+\(([^)]+)\)", content)
    pressure_match = re.search(r"pressure\s+([^;]+);", content)
    k_match = re.search(r"turbulentKE\s+([^;]+);", content)
    omega_match = re.search(r"turbulentOmega\s+([^;]+);", content)

    # Check if all required values were found
    if not velocity_match:
        raise ValueError("flowVelocity not found in initialConditions file")
    if not pressure_match:
        raise ValueError("pressure not found in initialConditions file")
    if not k_match:
        raise ValueError("turbulentKE not found in initialConditions file")
    if not omega_match:
        raise ValueError("turbulentOmega not found in initialConditions file")

    # Extract and convert values
    velocity = np.array([float(x) for x in velocity_match.group(1).split()])
    pressure = float(pressure_match.group(1))
    k = float(k_match.group(1))
    omega = float(omega_match.group(1))

    return {
        "p": np.array(pressure),
        "U": velocity,
        "k": np.array(k),
        "omega": np.array(omega),
    }
