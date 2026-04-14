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
Hydra entrypoint for config-driven benchmark evaluation.

Run from this directory::

    python main.py
    python main.py --config-name=config_volume
    python main.py case_id=run_1 run.device=cuda:0
    python main.py 'case_id=[run_1,run_11]'

Default Hydra config is ``conf/config_surface.yaml``. Use
``--config-name=config_volume`` for the volume benchmark. Same layout as
``workflows/domino_design_sensitivities/`` (Hydra + ``conf/``).
"""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig

from physicsnemo.cfd.evaluation.benchmarks.engine import BenchmarkPolicyError, run_benchmark
from physicsnemo.cfd.evaluation.benchmarks.hydra_utils import hydra_config_to_benchmark_dict
from physicsnemo.cfd.evaluation.config import Config


@hydra.main(version_base="1.3", config_path="conf", config_name="config_surface")
def main(cfg: DictConfig) -> None:
    """
    Load benchmark configuration with Hydra/OmegaConf interpolation and run.

    Parameters
    ----------
    cfg : DictConfig
        Composed user config (``conf/config_surface.yaml`` by default, or
        ``config_volume`` via ``--config-name=config_volume``) plus CLI overrides.
    """
    raw, case_id = hydra_config_to_benchmark_dict(cfg)
    config = Config.from_dict(raw)
    try:
        results = run_benchmark(config, case_id=case_id)
    except BenchmarkPolicyError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(f"Completed {len(results)} run(s). Results in {config.run.output_dir}")


if __name__ == "__main__":
    main()
