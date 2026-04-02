"""Shared helpers for inference model wrappers (VTK/STL, datapipe tensors, etc.)."""

from physicsnemo.cfd.evaluation.inference.common_wrapper_utils.vtk_datapipe_io import (
    build_surface_data_dict,
    build_volume_data_dict,
    read_stl_geometry,
    read_surface_from_vtp,
    read_volume_from_vtu,
    run_id_from_case_id,
)

__all__ = [
    "build_surface_data_dict",
    "build_volume_data_dict",
    "read_stl_geometry",
    "read_surface_from_vtp",
    "read_volume_from_vtu",
    "run_id_from_case_id",
]
