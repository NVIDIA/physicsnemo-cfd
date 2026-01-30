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

import pyvista as pv
import numpy as np
from pathlib import Path
from physicsnemo.cfd.hybrid_initialization_tools.utilities.openfoam_utils import (
    read_openfoam_mesh,
    read_openfoam_cell_centers,
    replace_internal_field,
    interpolate_fields,
    parse_initial_conditions,
)
from physicsnemo.cfd.hybrid_initialization_tools.utilities.pyvista_utils import (
    mask_pointset,
    add_points_to_pointset,
    replace_pointset_array_names,
)
from itertools import product

try:
    pwd = Path(__file__).parent
except NameError:
    pwd = Path(".").absolute()  # Fallback for interactive use

print("Loading mesh w/ potential flow initialization (same mesh as simulation mesh)...")
mesh_pf = read_openfoam_mesh(pwd)

print("Cleaning up the potential flow initialization...")
# Destroy all point information, since we won't use it
mesh_pf.point_data.clear()

print("Getting cell centers for simulation mesh...")
cell_centers: np.ndarray = mesh_pf.cell_centers().points

print("Loading mesh w/ ML initialization (different mesh than simulation mesh)...")
mesh_ml = pv.read(pwd / "from_domino" / "predicted_flow.vtu")

print("Cleaning up the ML initialization...")
# Destroy all cell information, connectivity, and unused field information, since we won't use it
points_ml: pv.PointSet = replace_pointset_array_names(
    pointset=mesh_ml.cast_to_pointset(),
    mapping={
        "UMeanTrimPred": "U",
        "pMeanTrimPred": "p",
        "TKEPred": "k",
        "nutMeanTrimPred": "nut",
    },
)

# Trim out points outside of DoMINO's prediction domain
bad_points = (points_ml["p"] == 0) & np.all(points_ml["U"] == 0, axis=1)
print(
    f"Eliminating {np.sum(bad_points)} points outside of DoMINO's prediction domain..."
)
points_ml = mask_pointset(points_ml, np.logical_not(bad_points))

# Add in boundary points, to allow smoother interpolation

# Get the freestream conditions from initialConditions file
conditions = parse_initial_conditions(pwd / "from_domino" / "initialConditions")
conditions.pop("omega")
conditions["nut"] = (
    0.1 * 1.5e-5
)  # Asymptote to turbulent viscosity ratio ~0.1 in the far-field

# Generate all 8 vertices of the bounding box
bounds = mesh_pf.bounds  # Returns [xmin, xmax, ymin, ymax, zmin, zmax]
boundary_points = {}
for x_high, y_high, z_high in product([False, True], repeat=3):
    # Convert boolean to index: False->min (0,2,4), True->max (1,3,5)
    x_idx = 1 if x_high else 0  # 0 for xmin, 1 for xmax
    y_idx = 3 if y_high else 2  # 2 for ymin, 3 for ymax
    z_idx = 5 if z_high else 4  # 4 for zmin, 5 for zmax
    # Create the vertex coordinate
    vertex = (bounds[x_idx], bounds[y_idx], bounds[z_idx])
    boundary_points[vertex] = conditions

points_ml = add_points_to_pointset(
    points_ml,
    new_points=np.array(list(boundary_points.keys())),
    new_data={
        field: np.stack([v[field] for v in boundary_points.values()])
        for field in conditions.keys()
    },
)

print("Interpolating ML initialization onto simulation mesh...")
interpolated_fields: dict[str, np.ndarray] = interpolate_fields(
    original_coords=points_ml.points,
    target_coords=cell_centers,
    fields=dict(points_ml.point_data),
)

print("Combining potential flow and ML initializations...")
combined_mesh = mesh_pf.copy()
for field, data in combined_mesh.cell_data.items():
    combined_mesh.cell_data[f"{field}_pf"] = combined_mesh.cell_data.pop(field)
combined_mesh.cell_data.update(
    {f"{field}_ml": data for field, data in interpolated_fields.items()}
)

print("Writing combined VTU file...")
combined_mesh.save(pwd / "combined_mesh.vtu")

print("Done!")
