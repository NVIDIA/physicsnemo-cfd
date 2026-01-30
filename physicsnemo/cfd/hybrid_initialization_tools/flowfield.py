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

from dataclasses import dataclass
from typing import Literal, Sequence
import pyvista as pv
from pathlib import Path
import warnings


@dataclass
class Flowfield:
    """Represents a computational fluid dynamics flow field with associated mesh and field data.

    This class encapsulates a PyVista mesh along with the names of key flow field variables
    (velocity, pressure, turbulent kinetic energy) to provide a unified interface for
    working with CFD data from various sources.

    Attributes
    ----------
    mesh : pv.PolyData
        PyVista mesh containing the geometry and field data.
    velocity_fieldname : str
        Name of the velocity field in the mesh data.
    pressure_fieldname : str
        Name of the pressure field in the mesh data.
    k_fieldname : str
        Name of the turbulent kinetic energy field in the mesh data.
    omega_fieldname : str
        Name of the specific dissipation rate field in the mesh data.
    """

    mesh: pv.PolyData
    velocity_fieldname: str = "U"
    pressure_fieldname: str = "p"
    k_fieldname: str = "k"
    omega_fieldname: str = "omega"

    @classmethod
    def from_vtu(
        cls,
        path: Path,
        velocity_fieldname: str | None = None,
        pressure_fieldname: str | None = None,
        k_fieldname: str | None = None,
        omega_fieldname: str | None = None,
    ) -> "Flowfield":
        """Create a Flowfield instance from a VTU file.

        This method reads a VTU file and attempts to automatically identify the velocity,
        pressure, and turbulent kinetic energy fields if not explicitly provided.

        Parameters
        ----------
        path : Path
            Path to the VTU file.
        velocity_fieldname : str, optional
            Name of the velocity field in the VTU file. If None, will attempt to automatically determine from common names.
        pressure_fieldname : str, optional
            Name of the pressure field in the VTU file. If None, will attempt to automatically determine from common names.
        k_fieldname : str, optional
            Name of the turbulent kinetic energy field in the VTU file. If None, will attempt to automatically determine from common names.
        omega_fieldname : str, optional
            Name of the specific dissipation rate field in the VTU file. If None, will attempt to automatically determine from common names.

        Returns
        -------
        Flowfield
            A new Flowfield instance containing the mesh data with properly mapped field names.

        Raises
        ------
        ValueError
            If any required field name cannot be automatically determined and is not provided explicitly.
        """
        mesh = pv.read(path)

        # Try to automatically parse the fieldnames. Values are None if no match can be found to the provided literals.
        auto_parsed_fieldnames: dict[str, str | None] = cls.parse_fieldnames(
            mesh,
            location="cell",
            warn_on_no_match=False,
        )

        # Determine the velocity fieldname, with some fallbacks
        if velocity_fieldname is None:
            if auto_parsed_fieldnames["velocity"] is not None:
                velocity_fieldname = auto_parsed_fieldnames["velocity"]
            else:
                raise ValueError(
                    "`velocity_fieldname` was not provided and could not be automatically determined from the mesh.\n"
                    "Please provide it.\n"
                    f"Available fields: {mesh.cell_data.keys()=}"
                )
        if pressure_fieldname is None:
            if auto_parsed_fieldnames["pressure"] is not None:
                pressure_fieldname = auto_parsed_fieldnames["pressure"]
            else:
                raise ValueError(
                    "`pressure_fieldname` was not provided and could not be automatically determined from the mesh.\n"
                    "Please provide it.\n"
                    f"Available fields: {mesh.cell_data.keys()=}"
                )
        if k_fieldname is None:
            if auto_parsed_fieldnames["k"] is not None:
                k_fieldname = auto_parsed_fieldnames["k"]
            else:
                raise ValueError(
                    "`k_fieldname` was not provided and could not be automatically determined from the mesh.\n"
                    "Please provide it.\n"
                    f"Available fields: {mesh.cell_data.keys()=}"
                )
        if omega_fieldname is None:
            if auto_parsed_fieldnames["omega"] is not None:
                omega_fieldname = auto_parsed_fieldnames["omega"]
            else:
                raise ValueError(
                    "`omega_fieldname` was not provided and could not be automatically determined from the mesh.\n"
                    "Please provide it.\n"
                    f"Available fields: {mesh.cell_data.keys()=}"
                )

        return cls(
            mesh=mesh,
            velocity_fieldname=velocity_fieldname,
            pressure_fieldname=pressure_fieldname,
            k_fieldname=k_fieldname,
            omega_fieldname=omega_fieldname,
        )

    @staticmethod
    def parse_fieldnames(
        mesh: pv.PolyData,
        location: Literal["cell", "point"] = "cell",
        velocity_literals: Sequence[str] = ("Velocity", "U", "Vel"),
        pressure_literals: Sequence[str] = ("Pressure", "p"),
        k_literals: Sequence[str] = (
            "k",
            "TKE",
            "Turbulent Energy",
            "Turbulent Kinetic Energy",
        ),
        omega_literals: Sequence[str] = (
            "omega",
            "Specific Dissipation Rate",
            "omega_t",
        ),
        nut_literals: Sequence[str] = (
            "nut",
            "nutilde",
            "nu_t",
            "nu_tilde",
            "Turbulent Viscosity",
        ),
        warn_on_unexpected_array_shapes: bool = True,
        warn_on_no_match: bool = True,
    ) -> dict[Literal["velocity", "pressure", "k", "omega", "nut"], str | None]:
        """Parse field names from a mesh by matching against common naming conventions.

        This function attempts to identify velocity, pressure, and turbulent kinetic energy
        fields in the mesh data by matching against common naming patterns. It first tries
        exact matches (case-insensitive) and then falls back to prefix matching.

        Parameters
        ----------
        mesh : pv.PolyData
            The mesh containing the field data.
        location : {'cell', 'point'}, optional
            Whether to look in cell or point data.
        velocity_literals : Sequence[str], optional
            Possible names for velocity fields.
        pressure_literals : Sequence[str], optional
            Possible names for pressure fields.
        k_literals : Sequence[str], optional
            Possible names for turbulent kinetic energy fields.
        omega_literals : Sequence[str], optional
            Possible names for specific dissipation rate fields.
        nut_literals : Sequence[str], optional
            Possible names for turbulent viscosity fields.
        warn_on_unexpected_array_shapes : bool, optional
            Whether to warn about arrays with unexpected shapes.
        warn_on_no_match : bool, optional
            Whether to warn when no matching field is found.

        Returns
        -------
        dict
            Dictionary mapping field types to their identified field names. Values will be None if no match is found.
        """
        if location == "cell":
            existing_fieldnames = mesh.cell_data.keys()
        elif location == "point":
            existing_fieldnames = mesh.point_data.keys()
        else:
            raise ValueError(f"Invalid location: {location}")

        # Split the fieldnames into scalar and vector fields
        scalar_existing_fieldnames = []
        vector_existing_fieldnames = []
        for name in existing_fieldnames:
            array = mesh.get_array(name)
            if array.ndim == 1:  # Shape: (n,)
                scalar_existing_fieldnames.append(name)
            elif array.ndim == 2 and array.shape[1] == 3:  # Shape: (n, 3)
                vector_existing_fieldnames.append(name)
            else:
                if warn_on_unexpected_array_shapes:
                    warnings.warn(
                        f"Unexpected array shape: {array.shape} for fieldname: {name}.\nExpected either (n,) or (n, 3) for scalar and vector fields, respectively."
                    )

        def parse_field(
            literals: Sequence[str],
            type: Literal["vector", "scalar"],
        ) -> str | None:
            """
            Parse field names from available fieldnames based on literal matches.

            Parameters
            ----------
            literals : Sequence[str]
                Possible names to match against.
            type : {'vector', 'scalar'}
                Whether to search in vector or scalar fields.

            Returns
            -------
            str or None
                The matched field name, or None if no match found.
            """
            if type == "vector":
                fieldnames = vector_existing_fieldnames
            elif type == "scalar":
                fieldnames = scalar_existing_fieldnames
            else:
                raise ValueError(f"Invalid type: {type}")

            # Try a literal match first
            for literal in literals:
                for fieldname in fieldnames:
                    if literal.lower().strip() == fieldname.lower().strip():
                        return fieldname

            # If no literal match, match if it starts with the literal
            possible_matches = []
            for literal in literals:
                for fieldname in fieldnames:
                    if fieldname.lower().strip().startswith(literal.lower().strip()):
                        possible_matches.append(fieldname)

            if len(possible_matches) == 1:
                return possible_matches[0]
            elif len(possible_matches) > 1:
                if warn_on_no_match:
                    warnings.warn(
                        f"Found multiple {type=} fields on {location=} partial-matching {literals=}.\n"
                        f"Available fields that match these criteria: {possible_matches=}\n"
                        f"Available {type} fields on {location}: {fieldnames=}"
                    )
                return None

            # If still no match, return None
            if warn_on_no_match:
                warnings.warn(
                    f"Did not find a {type=} field on {location=} matching any of {literals=}.\n"
                    f"Available {type} fields on {location}: {fieldnames=}"
                )
            return None

        velocity_fieldname = parse_field(velocity_literals, "vector")
        pressure_fieldname = parse_field(pressure_literals, "scalar")
        k_fieldname = parse_field(k_literals, "scalar")
        omega_fieldname = parse_field(omega_literals, "scalar")
        nut_fieldname = parse_field(nut_literals, "scalar")

        return {
            "velocity": velocity_fieldname,
            "pressure": pressure_fieldname,
            "k": k_fieldname,
            "omega": omega_fieldname,
            "nut": nut_fieldname,
        }
