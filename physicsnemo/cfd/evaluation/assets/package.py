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

"""Remote and local model asset roots (Hugging Face Hub, local dirs, S3). Not an earth2studio dependency."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

HF_PREFIX = "hf://"
S3_PREFIX = "s3://"
NGC_PREFIX = "ngc://"
FILE_PREFIX = "file://"

_HF_URI_RE = re.compile(
    r"^hf://(?P<repo>[^@]+?)(?:@(?P<rev>[^/]+))?$",
)


def _parse_hf_uri(root: str) -> tuple[str, str]:
    """Return (repo_id, revision) from ``hf://org/name@rev``."""
    m = _HF_URI_RE.match(root.strip())
    if not m:
        raise ValueError(
            f"Invalid hf:// URI: {root!r} (expected hf://org/repo@revision)"
        )
    repo = m.group("repo").strip("/")
    rev = (m.group("rev") or "main").strip()
    return repo, rev


class Package:
    """
    Resolve paths inside a package root: ``hf://``, ``s3://``, ``file://`` or local directory.

    Parameters
    ----------
    root : str
        Root URI or absolute path.
    cache_options : dict, optional
        For HF: optional ``cache_storage`` directory (defaults to :meth:`default_cache` / ``hf``).
    """

    def __init__(
        self,
        root: str,
        *,
        cache_options: dict[str, Any] | None = None,
    ) -> None:
        self.root = root.rstrip("/")
        self.cache_options = dict(cache_options or {})

    @classmethod
    def default_cache(cls, subpath: str = "") -> str:
        """Default under ``~/.cache/physicsnemo-cfd/models`` or ``PHYSICSNEMO_CFD_MODEL_CACHE``."""
        base = os.environ.get(
            "PHYSICSNEMO_CFD_MODEL_CACHE",
            os.path.join(
                os.path.expanduser("~"), ".cache", "physicsnemo-cfd", "models"
            ),
        )
        path = os.path.join(base, subpath) if subpath else base
        os.makedirs(path, exist_ok=True)
        return path

    def resolve(self, file_path: str) -> str:
        """Download/cache if needed and return a local filesystem path to the asset."""
        rel = file_path.lstrip("/")
        if self.root.startswith(HF_PREFIX):
            return self._resolve_hf(rel)
        if self.root.startswith(S3_PREFIX):
            return self._resolve_s3(rel)
        if self.root.startswith(NGC_PREFIX):
            raise NotImplementedError(
                "ngc:// model packages are not implemented yet; use local checkpoint paths or hf://."
            )
        return self._resolve_local(rel)

    def _resolve_local(self, rel: str) -> str:
        raw = self.root
        if raw.startswith(FILE_PREFIX):
            raw = raw[len(FILE_PREFIX) :]
        base = Path(raw).expanduser().resolve()
        local = (base / rel).resolve()
        try:
            local.relative_to(base)
        except ValueError as e:
            raise ValueError(f"Asset path {rel!r} escapes package root {base}") from e
        if not local.exists():
            raise FileNotFoundError(f"Asset not found in local package: {local}")
        return str(local)

    def _hf_cache_dir(self) -> str:
        return str(self.cache_options.get("cache_storage") or self.default_cache("hf"))

    def _resolve_hf(self, rel: str) -> str:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import (
                EntryNotFoundError,
                LocalEntryNotFoundError,
                RepositoryNotFoundError,
                RevisionNotFoundError,
            )
        except ImportError as e:
            raise ImportError(
                "hf:// packages require huggingface_hub. Install: "
                "pip install 'nvidia-physicsnemo-cfd[evaluation-hf]'"
            ) from e

        _hf_fallback_errors = (
            EntryNotFoundError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

        repo_id, revision = _parse_hf_uri(self.root)
        cache_dir = self._hf_cache_dir()

        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=rel,
                revision=revision,
                cache_dir=cache_dir,
            )
        except _hf_fallback_errors:
            pass

        # Directory-style checkpoint: pull snapshot subset
        snap = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=[f"{rel}/**", f"{rel}/*", rel],
        )
        sub = os.path.join(snap, rel)
        if os.path.isdir(sub):
            return sub
        if os.path.isfile(sub):
            return sub
        raise FileNotFoundError(
            f"Could not resolve {rel!r} as file or directory in {repo_id}@{revision}"
        )

    def _resolve_s3(self, rel: str) -> str:
        try:
            import fsspec
        except ImportError as e:
            raise ImportError(
                "s3:// packages require fsspec. Install: pip install 'nvidia-physicsnemo-cfd[evaluation-hf]'"
            ) from e

        full_remote = f"{self.root}/{rel}"

        dest_root = Path(
            self.cache_options.get("cache_storage") or self.default_cache("s3")
        )
        key_hash = hashlib.sha256(full_remote.encode()).hexdigest()[:24]
        dest_dir = dest_root / key_hash
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / Path(rel).name

        if dest_file.is_file():
            return str(dest_file.resolve())

        try:
            import filelock
        except ImportError as e:
            raise ImportError(
                "Concurrent-safe s3:// package caches require filelock. Install: "
                "pip install 'nvidia-physicsnemo-cfd[evaluation-hf]'"
            ) from e

        lock_file = dest_file.with_suffix(".lock")

        # Multi-rank jobs (torchrun): coordinate so only one writer creates dest_file bytes.
        with filelock.FileLock(str(lock_file)):
            if dest_file.is_file():
                return str(dest_file.resolve())
            fs, rpath = fsspec.url_to_fs(full_remote)
            if fs.isfile(rpath):
                fs.get(rpath, str(dest_file))
                return str(dest_file.resolve())
        raise FileNotFoundError(f"s3 asset not found (file): {full_remote}")


def maybe_touch_hf_config_json(package: Package | None) -> None:
    """Best-effort Hub access for optional ``config.json`` (e.g. download analytics)."""
    if package is None:
        return
    if not package.root.startswith(HF_PREFIX):
        return
    try:
        package.resolve("config.json")
    except FileNotFoundError:
        pass
