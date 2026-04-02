"""Mesh I/O and normalization statistics loading."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch


def load_global_stats(stats_path: str, device: str = "cpu") -> dict[str, Any]:
    """Load normalization statistics from JSON (mean / std_dev keys).

    Returns dict with "mean" and "std" (std_dev in file) as tensors on device.
    """
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    with open(path, "r") as f:
        data = json.load(f)
    return {
        "mean": {
            k: torch.tensor(v, device=device, dtype=torch.float32)
            for k, v in data["mean"].items()
        },
        "std": {
            k: torch.tensor(v, device=device, dtype=torch.float32)
            for k, v in data["std_dev"].items()
        },
    }


def surface_factors_from_global_stats(
    data: dict[str, Any],
    device: str,
) -> dict[str, torch.Tensor]:
    """Build ``mean`` / ``std`` vectors for TransolverDataPipe from ``global_stats.json``.

    Stacks pressure (1) and shear_stress (3) using ``mean`` / ``std_dev`` in the JSON
    (same layout as ``surface_fields_normalization.npz`` from training).
    """
    mp = np.asarray(data["mean"]["pressure"], dtype=np.float64).ravel()
    mt = np.asarray(data["mean"]["shear_stress"], dtype=np.float64).ravel()
    sp = np.asarray(data["std_dev"]["pressure"], dtype=np.float64).ravel()
    st = np.asarray(data["std_dev"]["shear_stress"], dtype=np.float64).ravel()
    mean = np.concatenate([mp[:1], mt[:3]])
    std = np.concatenate([sp[:1], st[:3]])
    return {
        "mean": torch.tensor(mean, device=device, dtype=torch.float32),
        "std": torch.tensor(std, device=device, dtype=torch.float32),
    }


def volume_factors_from_global_stats(
    data: dict[str, Any],
    device: str,
) -> dict[str, torch.Tensor]:
    """Build ``mean`` / ``std`` vectors for volume ``TransolverDataPipe`` from ``global_stats.json``.

    Channel order matches training / ``volume_fields_normalization.npz``:
    **velocity (3)**, **pressure_volume** (1), **turbulent_viscosity** (1).

    Uses keys ``velocity``, ``pressure_volume`` (or legacy ``pressure`` for volume pressure),
    and ``turbulent_viscosity`` under ``mean`` / ``std_dev``.
    """
    mean_block = data["mean"]
    std_block = data["std_dev"]
    pkey = "pressure_volume" if "pressure_volume" in mean_block else "pressure"
    if pkey not in mean_block:
        raise KeyError(
            "global_stats.json must include mean/std for volume pressure as "
            "'pressure_volume' or 'pressure'"
        )
    for key in ("velocity", "turbulent_viscosity"):
        if key not in mean_block:
            raise KeyError(f"global_stats.json missing 'mean' entry for {key!r}")

    mv = np.asarray(mean_block["velocity"], dtype=np.float64).ravel()
    mp = np.asarray(mean_block[pkey], dtype=np.float64).ravel()
    mn = np.asarray(mean_block["turbulent_viscosity"], dtype=np.float64).ravel()
    sv = np.asarray(std_block["velocity"], dtype=np.float64).ravel()
    sp = np.asarray(std_block[pkey], dtype=np.float64).ravel()
    sn = np.asarray(std_block["turbulent_viscosity"], dtype=np.float64).ravel()

    mean = np.concatenate([mv[:3], mp[:1], mn[:1]])
    std = np.concatenate([sv[:3], sp[:1], sn[:1]])
    return {
        "mean": torch.tensor(mean, device=device, dtype=torch.float32),
        "std": torch.tensor(std, device=device, dtype=torch.float32),
    }


def resolve_global_stats_path(stats_path: str | Path) -> Path:
    """Resolve ``global_stats.json`` from config ``stats_path`` (file or directory)."""
    p = Path(stats_path)
    if p.is_file() and p.name == "global_stats.json":
        return p
    if p.is_file():
        return p.parent / "global_stats.json"
    return p / "global_stats.json"


def load_transolver_surface_factors(
    stats_path: str,
    device: str,
) -> dict[str, torch.Tensor] | None:
    """Load surface normalization for ``TransolverDataPipe``: prefer ``global_stats.json``, else npz."""
    gs_path = resolve_global_stats_path(stats_path)
    if gs_path.exists():
        with open(gs_path) as f:
            data = json.load(f)
        return surface_factors_from_global_stats(data, device)

    p = Path(stats_path)
    norm_dir = p.parent if p.is_file() else (p if p.is_dir() else p.parent)
    npz_path = norm_dir / "surface_fields_normalization.npz"
    if npz_path.exists():
        norm_data = np.load(str(npz_path))
        return {
            "mean": torch.from_numpy(norm_data["mean"]).to(device),
            "std": torch.from_numpy(norm_data["std"]).to(device),
        }
    return None


def load_transolver_volume_factors(
    stats_path: str,
    device: str,
) -> dict[str, torch.Tensor] | None:
    """Load volume normalization for ``TransolverDataPipe``.

    Prefer ``global_stats.json`` (same layout as surface), else ``volume_fields_normalization.npz``.
    """
    gs_path = resolve_global_stats_path(stats_path)
    if gs_path.exists():
        with open(gs_path) as f:
            data = json.load(f)
        try:
            return volume_factors_from_global_stats(data, device)
        except KeyError:
            # JSON present but not a volume stats file (e.g. surface-only global_stats).
            pass

    p = Path(stats_path)
    norm_dir = p.parent if p.is_file() else (p if p.is_dir() else p.parent)
    npz_path = norm_dir / "volume_fields_normalization.npz"
    if npz_path.exists():
        norm_data = np.load(str(npz_path))
        return {
            "mean": torch.from_numpy(norm_data["mean"]).to(device),
            "std": torch.from_numpy(norm_data["std"]).to(device),
        }
    return None


def load_mesh(mesh_path: str) -> pv.PolyData:
    """Load a VTP (or other PyVista-readable) mesh from disk."""
    path = Path(mesh_path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.PolyData):
        raise TypeError(f"Expected PolyData, got {type(mesh)}")
    return mesh
