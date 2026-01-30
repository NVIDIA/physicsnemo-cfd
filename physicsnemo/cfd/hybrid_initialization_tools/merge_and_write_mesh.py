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
    read_openfoam_cell_centers,
    replace_internal_field,
    interpolate_fields,
)
from physicsnemo.cfd.hybrid_initialization_tools.utilities.pyvista_utils import (
    mask_pointset,
    add_points_to_pointset,
    replace_pointset_array_names,
)

try:
    pwd = Path(__file__).parent
except NameError:
    pwd = Path(".").absolute()  # Fallback for interactive use

print("Reading combined mesh...")
combined_mesh = pv.read(pwd / "combined_mesh.vtu")

print("Adding field for omega_ml, inferred from k_ml and nut_ml...")
combined_mesh["omega_ml"] = (
    np.maximum(combined_mesh["k_ml"], 0.24)
    # * 1.225 # OpenFOAM-assumed density value
    / np.maximum(combined_mesh["nut_ml"], 1.507e-5)
)

print("Merging mesh...")
merged_mesh = pv.UnstructuredGrid(combined_mesh)
merged_mesh.cell_data.clear()

### Create merge function
k_freestream = 0.24
k_lower_threshold = 1.5 * k_freestream
k_upper_threshold = 3 * k_freestream

ml_weighting = (
    np.sin(
        np.pi
        / 2
        * np.clip(
            (combined_mesh.cell_data["k_ml"] - k_lower_threshold)
            / (k_upper_threshold - k_lower_threshold),
            0,
            1,
        )
    )
    ** 2
)

merged_mesh.cell_data["ml_weighting"] = ml_weighting
for field in ["U", "p", "k", "omega"]:
    merged_mesh[field] = np.einsum(
        "i...,i->i...", combined_mesh[f"{field}_ml"], ml_weighting
    ) + np.einsum("i...,i->i...", combined_mesh[f"{field}_pf"], 1 - ml_weighting)
print("Writing initialization for OpenFOAM...")
replace_internal_field(pwd / "0_org/U", pwd / "0/U", merged_mesh.cell_data["U"])
print("U written.")
replace_internal_field(pwd / "0_org/p", pwd / "0/p", merged_mesh.cell_data["p"])
print("p written.")
replace_internal_field(pwd / "0_org/k", pwd / "0/k", merged_mesh.cell_data["k"])
print("k written.")
replace_internal_field(
    pwd / "0_org/omega", pwd / "0/omega", merged_mesh.cell_data["omega"]
)
print("omega written.")

print("Done!")
