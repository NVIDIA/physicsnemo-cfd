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

"""CFD prediction containers with independent member/domain parallelism.

Requires PhysicsNeMo tensor-mode statistics. Run:
    torchrun --standalone --nproc-per-node=4 workflows/benchmarking/uq_member_parallel.py

Analytic decoded pressure distributions make this example self-checking. Replace
``decoded_pressure`` with a model adapter that decodes onto the same point ordering.
This is an application-owned same-case workflow, separate from the benchmark
engine's existing case-parallel scheduler.
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard

from physicsnemo.cfd.evaluation.datasets.schema import build_predictive_distribution
from physicsnemo.domain_parallel import ShardTensor, register_custom_ops
from physicsnemo.metrics.general.ensemble_metrics import Mean, Variance


def decoded_pressure(member_ids, point_ids):
    """Decode an analytic GP-like prediction: pressure and marginal variance in Pa squared (means in Pa)."""
    normalized_mean = member_ids[:, None] * 0.3 + point_ids[None, :] * 0.05
    normalized_std = 0.1 + member_ids[:, None] * 0.02
    mean = 100.0 + 20.0 * normalized_mean
    variance = (20.0 * normalized_std).square().expand_as(mean)
    return mean, variance


@torch.no_grad()
def main():
    """Check field moments, total GP variance, an observable, and coverage metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-groups", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Auto uses CUDA only when every local rank has a visible GPU; otherwise CPU.",
    )
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    enough_gpus = visible_gpus >= local_world_size and local_rank < visible_gpus
    if args.device == "cuda" and not enough_gpus:
        parser.error(
            f"--device cuda requires one visible GPU per local rank: "
            f"{visible_gpus} GPUs available for {local_world_size} ranks. "
            "Use --device cpu or reduce --nproc-per-node."
        )
    use_cuda = args.device != "cpu" and enough_gpus
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    else:
        # CUDA imports register these automatically; CPU runs need the same handlers.
        register_custom_ops()
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    space_groups = world_size // args.member_groups
    if world_size not in (2, 4) or space_groups not in (1, 2):
        raise ValueError(
            "Use two ranks with one/two member groups, or four ranks with two"
        )
    mesh = init_device_mesh(
        device.type,
        [args.member_groups, space_groups],
        mesh_dim_names=["member", "space"],
    )
    member_rank, space_rank = mesh.get_coordinate()
    assignments = [[0, 1, 2, 3, 4]] if args.member_groups == 1 else [[0, 2, 4], [1, 3]]
    spatial_sizes = [10] if space_groups == 1 else [3, 7]
    nx = spatial_sizes[space_rank]
    start = sum(spatial_sizes[:space_rank])
    point_ids = torch.arange(start, start + nx, device=device, dtype=torch.float64)
    local_weights = (point_ids + 1).unsqueeze(0) * 0.01
    weights = ShardTensor.from_local(
        local_weights,
        mesh,
        [Replicate(), Shard(1)],
        sharding_shapes={1: [(1, n) for n in spatial_sizes]},
    )
    between = Variance(mode="tensor", correction=0)
    within = Mean(mode="tensor")
    force = Variance(mode="tensor", correction=0)

    # All ranks execute every round. A group with no remaining member contributes
    # an empty shard; it still participates in the matching statistics collectives.
    for step in range(max(map(len, assignments))):
        sizes = [int(step < len(ids)) for ids in assignments]
        assigned = assignments[member_rank][step : step + 1]
        member_ids = torch.tensor(assigned, device=device, dtype=torch.float64)
        mean, variance = decoded_pressure(member_ids, point_ids)
        shapes = {
            0: [(n, nx) for n in sizes],
            1: [(sizes[member_rank], n) for n in spatial_sizes],
        }
        mean = ShardTensor.from_local(
            mean,
            mesh,
            [Shard(0), Shard(1)],
            sharding_shapes=shapes,
        )
        variance = ShardTensor.from_local(
            variance,
            mesh,
            [Shard(0), Shard(1)],
            sharding_shapes=shapes,
        )
        between.update(mean, dim=0)
        within.update(variance, dim=0)
        # Domain reduction completes before moments of the integrated prediction.
        force.update((mean * weights).sum(1), dim=0)

    epistemic = between.finalize()
    aleatoric = within.finalize()
    sharded = build_predictive_distribution(
        mean=between.mean,
        std=(epistemic + aleatoric).sqrt(),
        epistemic_std=epistemic.sqrt(),
        aleatoric_std=aleatoric.sqrt(),
    )
    assert sharded.mean.placements == (Replicate(), Shard(0))
    assert between.count == within.count == force.count == 5

    def complete(value):
        """Gather final field summaries for the existing complete-field CFD metrics."""
        return value.full_tensor()

    # No prediction stack is gathered. The small stack below is only a reference
    # for this executable test, and is not needed in a production workflow.
    ids = torch.arange(5, device=device, dtype=torch.float64)
    points = torch.arange(10, device=device, dtype=torch.float64)
    means, variances = decoded_pressure(ids, points)
    expected_mean = means.mean(0)
    expected_epi = means.var(0, correction=0)
    expected_ale = variances.mean(0)
    torch.testing.assert_close(complete(sharded.mean), expected_mean)
    torch.testing.assert_close(
        complete(sharded.std).square(), expected_epi + expected_ale
    )
    torch.testing.assert_close(complete(sharded.epistemic_std).square(), expected_epi)
    torch.testing.assert_close(complete(sharded.aleatoric_std).square(), expected_ale)
    expected_force = (means * ((points + 1) * 0.01)).sum(1)
    torch.testing.assert_close(force.mean.to_local(), expected_force.mean())
    torch.testing.assert_close(
        force.finalize().to_local(), expected_force.var(correction=0)
    )
    # This is variation of member mean forces. GP force variance additionally
    # requires joint samples/cross-point covariance; marginal variances do not suffice.

    complete_prediction = build_predictive_distribution(
        mean=complete(sharded.mean).cpu().numpy(),
        std=complete(sharded.std).cpu().numpy(),
    )
    import physicsnemo.cfd.evaluation.metrics  # noqa: F401
    from physicsnemo.cfd.postprocessing_tools.metric_registry import get_metric

    metric = get_metric("coverage_95", domain="surface")
    partial = metric.partial(
        {"pressure": expected_mean.cpu().numpy()}, {"pressure": complete_prediction}
    )
    result = metric.finalize(partial)
    assert all(np.isclose(value, 1.0) for value in result.values())

    if rank == 0:
        print("CFD member/domain fields, GP variance, observable, and metrics PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
