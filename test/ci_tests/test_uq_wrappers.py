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

"""UQ wrapper plumbing: checkpoint routing, block-partition coverage, field-name fallback,
registry routing, and the UQ contract class attributes."""

from __future__ import annotations

import pytest
import torch

from physicsnemo.cfd.evaluation.metrics.mesh_bridge import uq_std_field_names
from physicsnemo.cfd.evaluation.models.common_wrapper_utils.geotransolver_runtime import (
    make_forward_permutation,
    resolve_checkpoint_file,
)


# --------------------------------------------------------------------------------------------
# Checkpoint routing: a specific file (epoch-in-name) -> (dir, epoch); directory / bad name reject
# --------------------------------------------------------------------------------------------


def test_resolve_checkpoint_file_parses_epoch(tmp_path) -> None:
    """A specific checkpoint file name resolves to ``(directory, epoch)``."""
    ckpt = tmp_path / "GeoTransolver.0.30.mdlus"
    ckpt.write_bytes(b"")  # existence not required, but harmless
    directory, epoch = resolve_checkpoint_file(str(ckpt))
    assert directory == tmp_path and epoch == 30
    # Nonexistent path still resolves purely from the file name (no silent latest-epoch fallback).
    d2, e2 = resolve_checkpoint_file(str(tmp_path / "checkpoint.0.100.pt"))
    assert d2 == tmp_path and e2 == 100


def test_resolve_checkpoint_file_rejects_directory_and_epochless(tmp_path) -> None:
    """A directory, empty path, or epoch-less file name is rejected (no silent latest-epoch)."""
    with pytest.raises(ValueError):
        resolve_checkpoint_file(str(tmp_path))  # directory -> ambiguous epoch
    with pytest.raises(ValueError):
        resolve_checkpoint_file("")  # empty
    with pytest.raises(ValueError):
        resolve_checkpoint_file(
            str(tmp_path / "GeoTransolver.mdlus")
        )  # no epoch in name


# --------------------------------------------------------------------------------------------
# Block-partition permutation covers the WHOLE point cloud (no subsampling of samples)
# --------------------------------------------------------------------------------------------


def test_make_forward_permutation_covers_all_points() -> None:
    """The per-case forward permutation is a full bijection over all points (no dropped indices)."""
    n = 137
    batch = {"embeddings": torch.zeros(1, n, 3)}
    perm = make_forward_permutation(batch)
    assert perm.shape == (n,)
    # Every point index appears exactly once -> the forward predicts all points, order restorable.
    assert torch.equal(torch.sort(perm).values, torch.arange(n))


# --------------------------------------------------------------------------------------------
# Automatic uncertainty field-name fallback (shared by mesh attachment and drag_uq)
# --------------------------------------------------------------------------------------------


def test_uq_std_field_names_fallback_and_override() -> None:
    """Std field names auto-derive from the prediction name unless explicitly overridden."""
    # No config -> derive from the prediction name (what comparison meshes attach by default).
    assert uq_std_field_names("pressure") == ("pressureStd", "pressureEpistemicStd")
    # Configured names win.
    assert uq_std_field_names("pressure", "pStd", "pEpi") == ("pStd", "pEpi")
    # Partial override: only the total std configured, epistemic still auto-derived.
    assert uq_std_field_names("wallShearStress", "wssStd", None) == (
        "wssStd",
        "wallShearStressEpistemicStd",
    )


# --------------------------------------------------------------------------------------------
# Registry routing + UQ contract attributes for the three example UQ wrappers
# --------------------------------------------------------------------------------------------


def test_uq_wrappers_registered_with_expected_contract() -> None:
    """The three UQ wrappers are registered and expose the expected SUPPORTS_UQ/UQ_METHOD contract."""
    # Importing the wrappers package registers every model wrapper.
    import physicsnemo.cfd.evaluation.models.wrappers  # noqa: F401
    from physicsnemo.cfd.evaluation.models.model_registry import get_model_wrapper

    gp = get_model_wrapper("geotransolver_gp_surface")
    mc = get_model_wrapper("geotransolver_mc_dropout_surface")
    ens = get_model_wrapper("geotransolver_ensemble_surface")

    # Analytic GP head: single-pass distribution.
    assert gp.SUPPORTS_UQ is True and gp.UQ_METHOD == "analytic"
    # Sampling wrappers: repeated-pass spread.
    assert mc.SUPPORTS_UQ is True and mc.UQ_METHOD == "sampling"
    assert ens.SUPPORTS_UQ is True and ens.UQ_METHOD == "sampling"
    # All three decorate physical-target (transformer_models) checkpoints -> no ρu² re-dim.
    for cls in (gp, mc, ens):
        assert cls.REDIMENSIONALIZE_OUTPUTS is False
