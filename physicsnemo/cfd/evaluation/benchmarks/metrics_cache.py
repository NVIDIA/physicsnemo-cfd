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

"""
On-disk cache for benchmark metric scalars.

Stores one JSON file per case containing float metrics only. Does not store
meshes, vector fields, or anything required for PNG/visual generation.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from pathlib import Path
from typing import Any

CACHE_FORMAT_VERSION = 1


def _canonical_fingerprint_path(path_str: str) -> str:
    """Normalize filesystem paths so equivalent locations share one cache fingerprint."""
    if not path_str:
        return ""
    p = Path(path_str).expanduser()
    try:
        return str(p.resolve(strict=False))
    except (OSError, RuntimeError):
        return str(p)


def _fingerprint_jsonify(obj: Any) -> Any:
    """
    Recursively convert values to JSON-stable Python scalars for hashing.

    Avoids fingerprint drift from NumPy scalar types and normalizes nested dicts.
    """

    if isinstance(obj, dict):
        return {str(k): _fingerprint_jsonify(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple)):
        return [_fingerprint_jsonify(v) for v in obj]
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real) and not isinstance(obj, bool):
        return float(obj)
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
        try:
            return _fingerprint_jsonify(obj.item())
        except Exception:
            pass
    return str(obj)


def _stable_json(obj: Any) -> str:
    """Serialize ``obj`` to a stable JSON string for hashing (sorted keys)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def metrics_cache_fingerprint(
    *,
    model_name: str,
    model_checkpoint: str,
    model_stats_path: str,
    model_kwargs: dict[str, Any],
    model_inference_domain: str | None,
    model_asset_identity: str | None = None,
    dataset_name: str,
    dataset_root: str,
    dataset_split: str | None,
    dataset_kwargs_resolved: dict[str, Any],
    output_dict: dict[str, Any],
    metric_specs: list[tuple[str, dict[str, Any]]],
) -> str:
    """
    Build a SHA-256 fingerprint for cache lookup and invalidation.

    Any change to model, dataset, output field maps, or metric definitions
    changes the digest so stale entries are not reused.

    Parameters
    ----------
    model_name : str
        Registered model name.
    model_checkpoint : str
        Checkpoint path.
    model_stats_path : str
        Normalization stats path.
    model_kwargs : dict
        Extra arguments passed to the model wrapper.
    model_inference_domain : str or None
        ``"surface"``, ``"volume"``, or ``None`` for registry default.
    model_asset_identity : str or None, optional
        Stable id when weights come from a remote :class:`~physicsnemo.cfd.evaluation.assets.Package`
        (e.g. ``package:hf://...|ckpt|stats``). When set, ``model_checkpoint`` / ``model_stats_path``
        in the payload are left empty so cache keys do not depend on local Hub cache paths.
    dataset_name : str
        Registered dataset adapter name.
    dataset_root : str
        Dataset root directory.
    dataset_split : str or None
        Dataset split, if any.
    dataset_kwargs_resolved : dict
        Adapter kwargs after alignment for this model.
    output_dict : dict
        Serializable ``OutputConfig`` mapping (see ``output_config_to_fingerprint_dict``).
    metric_specs : list of tuple
        Normalized ``(metric_name, kwargs)`` pairs from the benchmark config.

    Returns
    -------
    str
        64-character lowercase hex SHA-256 digest.
    """
    specs_serializable: list[Any] = []
    for name, kw in sorted(metric_specs, key=lambda x: (x[0], tuple(sorted(x[1].keys())))):
        specs_serializable.append(
            [name, _fingerprint_jsonify({k: kw[k] for k in sorted(kw.keys())})]
        )
    ck_fp = "" if model_asset_identity else _canonical_fingerprint_path(model_checkpoint)
    st_fp = "" if model_asset_identity else _canonical_fingerprint_path(model_stats_path)
    payload = {
        "fingerprint_schema": CACHE_FORMAT_VERSION,
        "model": {
            "name": model_name,
            "checkpoint": ck_fp,
            "stats_path": st_fp,
            "asset_identity": model_asset_identity or "",
            "kwargs": _fingerprint_jsonify(
                {k: model_kwargs[k] for k in sorted(model_kwargs.keys())}
            ),
            "inference_domain": model_inference_domain,
        },
        "dataset": {
            "name": dataset_name,
            "root": _canonical_fingerprint_path(dataset_root),
            "split": dataset_split,
            "kwargs": _fingerprint_jsonify(
                {k: dataset_kwargs_resolved[k] for k in sorted(dataset_kwargs_resolved.keys())}
            ),
        },
        "output": _fingerprint_jsonify(output_dict),
        "metrics": specs_serializable,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def case_id_cache_filename(case_id: str) -> str:
    """
    Map a case id to a safe JSON filename.

    Parameters
    ----------
    case_id : str
        Dataset case identifier.

    Returns
    -------
    str
        Filename of the form ``"<sanitized>.json"``.
    """
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in case_id)
    return f"{safe}.json"


def metrics_cache_file_path(cache_root: Path, fingerprint_hex: str, case_id: str) -> Path:
    """
    Path to the cache file for one case.

    Parameters
    ----------
    cache_root : Path
        Root metrics cache directory.
    fingerprint_hex : str
        Full fingerprint string (hex).
    case_id : str
        Case identifier.

    Returns
    -------
    Path
        ``cache_root / fingerprint_hex / <sanitized_case_id>.json``.
    """
    return Path(cache_root) / fingerprint_hex / case_id_cache_filename(case_id)


def output_config_to_fingerprint_dict(output: Any) -> dict[str, Any]:
    """
    Build an ordered dict of output field mappings for fingerprinting.

    Parameters
    ----------
    output : OutputConfig
        Evaluation output / VTK name mapping configuration.

    Returns
    -------
    dict
        Serializable mapping including mesh field names and streamlines key.
    """
    return {
        "mesh_field_names": {k: output.mesh_field_names[k] for k in sorted(output.mesh_field_names.keys())},
        "volume_mesh_field_names": {
            k: output.volume_mesh_field_names[k] for k in sorted(output.volume_mesh_field_names.keys())
        },
        "ground_truth_mesh_field_names": {
            k: output.ground_truth_mesh_field_names[k]
            for k in sorted(output.ground_truth_mesh_field_names.keys())
        },
        "ground_truth_volume_mesh_field_names": {
            k: output.ground_truth_volume_mesh_field_names[k]
            for k in sorted(output.ground_truth_volume_mesh_field_names.keys())
        },
        "streamlines_vector_canonical": output.streamlines_vector_canonical,
        "surface_interpolate_point_to_cell_for_metrics": output.surface_interpolate_point_to_cell_for_metrics,
        "surface_metrics_idw_k": output.surface_metrics_idw_k,
        "surface_metrics_idw_device": output.surface_metrics_idw_device,
    }


def _metrics_to_jsonable(case_metrics: dict[str, float]) -> dict[str, Any]:
    """Encode metric floats for JSON (NaN becomes null)."""
    out: dict[str, Any] = {}
    for k, v in case_metrics.items():
        if isinstance(v, float) and v != v:
            out[k] = None
        else:
            out[k] = v
    return out


def metrics_from_cache_json(metrics_obj: Any) -> dict[str, float] | None:
    """
    Parse the ``metrics`` object from a cache JSON file.

    Parameters
    ----------
    metrics_obj : Any
        Object loaded from JSON; must be a dict with float-coercible values or nulls.

    Returns
    -------
    dict or None
        Metric name to float value, or ``None`` if parsing fails.
    """
    if not isinstance(metrics_obj, dict):
        return None
    out: dict[str, float] = {}
    for k, v in metrics_obj.items():
        if v is None:
            out[str(k)] = float("nan")
        else:
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                return None
    return out


def read_metrics_cache(path: Path) -> dict[str, Any] | None:
    """
    Load and validate a cache file.

    Parameters
    ----------
    path : Path
        Path to a ``.json`` cache file.

    Returns
    -------
    dict or None
        Parsed payload if the file exists and matches the expected schema; otherwise ``None``.
    """
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("cache_format_version") != CACHE_FORMAT_VERSION:
        return None
    if "fingerprint" not in data or "metrics" not in data:
        return None
    return data


def write_metrics_cache(
    path: Path,
    *,
    fingerprint: str,
    model: str,
    dataset: str,
    case_id: str,
    case_metrics: dict[str, float],
    metric_dtype: str | None = None,
    comparison_mesh_path: str | None = None,
) -> None:
    """
    Atomically write a per-case cache entry (via a temporary file and replace).

    Parameters
    ----------
    path : Path
        Destination cache path (see ``metrics_cache_file_path``).
    fingerprint : str
        Full hex fingerprint for validation on read.
    model : str
        Model name.
    dataset : str
        Dataset adapter name.
    case_id : str
        Case identifier.
    case_metrics : dict
        Scalar metric name -> value.
    metric_dtype : str or None, optional
        Dtype string recorded when the comparison mesh was built, if any.
    comparison_mesh_path : str or None, optional
        Path written when ``reports.save_comparison_meshes`` saved a mesh for this case.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "model": model,
        "dataset": dataset,
        "case_id": case_id,
        "metrics": _metrics_to_jsonable(case_metrics),
        "metric_dtype": metric_dtype,
        "comparison_mesh_path": comparison_mesh_path,
    }
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def resolve_metrics_cache_root(*, enabled: bool, path: str, output_dir: str) -> Path | None:
    """
    Resolve the metrics cache root directory.

    Parameters
    ----------
    enabled : bool
        Whether caching is turned on in config.
    path : str
        User path, or empty to default under ``output_dir``.
    output_dir : str
        Benchmark ``run.output_dir`` (used when ``path`` is empty).

    Returns
    -------
    Path or None
        Absolute cache root, or ``None`` if caching is disabled.
    """
    if not enabled:
        return None
    p = (path or "").strip()
    if not p:
        return Path(output_dir).expanduser().resolve() / ".metrics_cache"
    return Path(p).expanduser().resolve()
