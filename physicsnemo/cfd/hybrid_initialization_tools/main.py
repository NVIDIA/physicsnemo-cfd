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

from pathlib import Path
from typing import Callable, Literal
import numpy as np
from physicsnemo.cfd.hybrid_initialization_tools.flowfield import Flowfield
from physicsnemo.cfd.hybrid_initialization_tools.utilities.openfoam_utils import (
    interpolate_fields,
)
from dataclasses import replace


def from_field_a_k(
    flowfield_a: Flowfield,
    flowfield_b: Flowfield,
) -> Flowfield:
    """
    An example blend strategy using the turbulent kinetic energy (k) field from flowfield_a.

    Returns a weight between 0 and 1, where 1 is all flowfield_a and 0 is all flowfield_b.

    Parameters
    ----------
    flowfield_a : Flowfield
        The first flowfield to blend.
    flowfield_b : Flowfield
        The second flowfield to blend.

    Returns
    -------
    Flowfield
        The blended flowfield.
    """
    field_a_k = flowfield_a.mesh.cell_data[flowfield_a.k_fieldname]
    k_freestream = 0.24
    k_lower_threshold = 1.5 * k_freestream
    k_upper_threshold = 3 * k_freestream

    weight = (
        np.sin(
            np.pi
            / 2
            * np.clip(
                (field_a_k - k_lower_threshold)
                / (k_upper_threshold - k_lower_threshold),
                0,
                1,
            )
        )
        ** 2
    )

    return weight


def create_hybrid_initialization(
    flowfield_a: Flowfield,
    flowfield_b: Flowfield,
    use_topology_from_mesh: Literal["a", "b"] = "a",
    flowfield_a_data_location: Literal["cell", "point"] = "cell",
    flowfield_b_data_location: Literal["cell", "point"] = "cell",
    blend_strategy: Callable[[Flowfield, Flowfield], Flowfield] = from_field_a_k,
    verbose: bool = True,
) -> Flowfield:
    """
    Creates a hybrid initialization by blending two flowfields.

    Parameters
    ----------
    flowfield_a : Flowfield
        The first flowfield to blend.
    flowfield_b : Flowfield
        The second flowfield to blend.
    use_topology_from_mesh : {'a', 'b'}, optional
        Specifies which mesh topology to use, by default 'a'.
    flowfield_a_data_location : {'cell', 'point'}, optional
        Data location for flowfield_a, by default 'cell'.
    flowfield_b_data_location : {'cell', 'point'}, optional
        Data location for flowfield_b, by default 'cell'.
    blend_strategy : Callable[[Flowfield, Flowfield], Flowfield], optional
        The strategy used to blend the flowfields, by default from_field_a_k.
    verbose : bool, optional
        If True, prints detailed information during processing, by default True.

    Returns
    -------
    Flowfield
        The resulting blended flowfield.
    """
    ### Validate arguments, to fail fast before long-running computations
    if use_topology_from_mesh not in ["a", "b"]:
        raise ValueError(f"Invalid use_topology_from_mesh: {use_topology_from_mesh=}")
    if flowfield_a_data_location not in ["cell", "point"]:
        raise ValueError(
            f"Invalid flowfield_a_data_location: {flowfield_a_data_location=}"
        )
    if flowfield_b_data_location not in ["cell", "point"]:
        raise ValueError(
            f"Invalid flowfield_b_data_location: {flowfield_b_data_location=}"
        )

    # # Handle the case where we want to use the topology of mesh b by swapping the arguments
    # if use_topology_from_mesh == "a":
    #     pass
    # elif use_topology_from_mesh == "b":
    #     return create_hybrid_initialization(
    #         flowfield_a=flowfield_b,
    #         flowfield_b=flowfield_a,
    #         use_topology_from_mesh="a",
    #         flowfield_a_data_location=flowfield_b_data_location,
    #         flowfield_b_data_location=flowfield_a_data_location,
    #         blend_strategy=lambda b, a: blend_strategy(a, b),
    #         verbose=verbose,
    #     )
    # else:
    #     raise ValueError(
    #         f"`use_topology_from_mesh` must be either 'a' or 'b', got {use_topology_from_mesh=}."
    #     )
    # # At this point, we want to use the topology of mesh a

    ### Interpolate fields
    if verbose:
        print("Computing data locations to compare meshes...")

    # Determine where the data is on each mesh (cell centers or points)
    if flowfield_a_data_location == "cell":
        coords_a = flowfield_a.mesh.cell_centers().points
    elif flowfield_a_data_location == "point":
        coords_a = flowfield_a.mesh.points

    if flowfield_b_data_location == "cell":
        coords_b = flowfield_b.mesh.cell_centers().points
    elif flowfield_b_data_location == "point":
        coords_b = flowfield_b.mesh.points

    ### Interpolation

    # Check if interpolation is needed
    if (
        flowfield_a_data_location == flowfield_b_data_location
        and coords_a.shape == coords_b.shape
        and np.allclose(coords_a, coords_b)
    ):
        if verbose:
            print("Fields are on the same mesh, no interpolation needed.")
        flowfield_a = replace(flowfield_a, mesh=flowfield_b.mesh)
        flowfield_b = replace(flowfield_b, mesh=flowfield_a.mesh)

    else:
        if use_topology_from_mesh == "a":
            if verbose:
                print("Interpolating fields from mesh b to mesh a...")
            interpolated_fields = interpolate_fields(
                original_coords=coords_b,
                target_coords=coords_a,
                fields=dict(
                    flowfield_b.mesh.cell_data
                    if flowfield_b_data_location == "cell"
                    else flowfield_b.mesh.point_data
                ),
            )
            mesh_b = flowfield_a.mesh.copy()
            mesh_b.cell_data.clear()
            mesh_b.point_data.clear()
            for fieldname, field in interpolated_fields.items():
                if flowfield_a_data_location == "cell":
                    mesh_b.cell_data[fieldname] = field
                elif flowfield_a_data_location == "point":
                    mesh_b.point_data[fieldname] = field
            flowfield_b = replace(flowfield_b, mesh=mesh_b)

        elif use_topology_from_mesh == "b":
            if verbose:
                print("Interpolating fields from mesh a to mesh b...")
            interpolated_fields = interpolate_fields(
                original_coords=coords_a,
                target_coords=coords_b,
                fields=dict(
                    flowfield_a.mesh.cell_data
                    if flowfield_a_data_location == "cell"
                    else flowfield_a.mesh.point_data
                ),
            )
            mesh_a = flowfield_b.mesh.copy()
            mesh_a.cell_data.clear()
            mesh_a.point_data.clear()
            for fieldname, field in interpolated_fields.items():
                if flowfield_b_data_location == "cell":
                    mesh_a.cell_data[fieldname] = field
                elif flowfield_b_data_location == "point":
                    mesh_a.point_data[fieldname] = field
            flowfield_a = replace(flowfield_a, mesh=mesh_a)

    # At this point, we have flowfield_a and flowfield_b on the same mesh

    ### Create the empty merged mesh
    if verbose:
        print("Creating empty merged mesh...")

    if use_topology_from_mesh == "a":
        merged_mesh = flowfield_a.mesh.copy()
        merged_mesh_data_location = flowfield_a_data_location
    elif use_topology_from_mesh == "b":
        merged_mesh = flowfield_b.mesh.copy()
        merged_mesh_data_location = flowfield_b_data_location
    merged_mesh.cell_data.clear()
    merged_mesh.point_data.clear()

    ### Interpolate meshes appropriately
    if verbose:
        print("Blending fields...")
    weight = blend_strategy(flowfield_a, flowfield_b)
    # A value of 1 corresponds to all flowfield_a, 0 corresponds to all flowfield_b

    fieldnames = ["U", "p", "k", "omega"]

    def get_field(
        field: Literal["a", "b"], fieldname: Literal["U", "p", "k", "omega"]
    ) -> np.ndarray:
        """Get a field from a flowfield.

        Parameters
        ----------
        field : Literal["a", "b"]
            Which flowfield to get the field from.
        fieldname : Literal["U", "p", "k", "omega"]
            The name of the field to get.

        Returns
        -------
        np.ndarray
            The field.
        """
        if field == "a":
            field = flowfield_a
        elif field == "b":
            field = flowfield_b

        if fieldname == "U":
            fieldname = field.velocity_fieldname
        elif fieldname == "p":
            fieldname = field.pressure_fieldname
        elif fieldname == "k":
            fieldname = field.k_fieldname
        elif fieldname == "omega":
            fieldname = field.omega_fieldname
        else:
            raise ValueError(f"Invalid fieldname: {fieldname=}")
        return (
            field.mesh.cell_data[fieldname]
            if merged_mesh_data_location == "cell"
            else field.mesh.point_data[fieldname]
        )

    if merged_mesh_data_location == "cell":
        for fieldname in fieldnames:
            merged_mesh.cell_data[fieldname] = np.einsum(
                "i...,i->i...", get_field("a", fieldname), weight
            ) + np.einsum("i...,i->i...", get_field("b", fieldname), 1 - weight)
            if verbose:
                print(f"Blended {fieldname=}.")
    elif merged_mesh_data_location == "point":
        for fieldname in fieldnames:
            merged_mesh.point_data[fieldname] = np.einsum(
                "i...,i->i...", get_field("a", fieldname), weight
            ) + np.einsum("i...,i->i...", get_field("b", fieldname), 1 - weight)
            if verbose:
                print(f"Blended {fieldname=}.")

    ### Return the new flowfield
    return Flowfield(
        mesh=merged_mesh,
        velocity_fieldname="U",
        pressure_fieldname="p",
        k_fieldname="k",
        omega_fieldname="omega",
    )


if __name__ == "__main__":
    from pathlib import Path
    import pyvista as pv

    example_dir = Path(__file__).parent.parent / "example_drivaerml_openfoam"
    flowfield_a = Flowfield.from_vtu(
        example_dir / "VTK" / "example_drivaerml_openfoam_0" / "internal.vtu"
    )

    mesh_ml = pv.read(example_dir / "from_domino" / "predicted_flow.vtu")
    flowfield_b = Flowfield(
        mesh=mesh_ml,
        velocity_fieldname="UMeanTrimPred",
        pressure_fieldname="pMeanTrimPred",
        k_fieldname="TKEPred",
        omega_fieldname="OmegaPred",
    )

    # flowfield
    result = create_hybrid_initialization(
        flowfield_a, flowfield_b, flowfield_b_data_location="point"
    )
