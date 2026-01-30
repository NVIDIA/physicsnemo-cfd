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

import numpy as np
from physicsnemo.sym.geometry.tessellation import Tessellation
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
import glob
import os
import re
from pathlib import Path
from multiprocessing import Pool
from itertools import product


def gen_pc(filename, pc_size):
    run_idx = re.search(r"\d+", filename).group()
    mesh = Tessellation.from_stl(filename, airtight=True)
    samples = mesh.sample_boundary(pc_size)
    output_filename = os.path.join(
        "./original_pointclouds/", f"input_pc_{str(int(pc_size))}_run_{run_idx}_final"
    )
    var_to_polyvtk(samples, output_filename)


if __name__ == "__main__":

    # fix numpy seed if needed
    np.random.seed(7)

    stl_files = glob.glob(os.path.join("original_stls", "*single_solid.stl"))
    Path("./original_pointclouds").mkdir(parents=True, exist_ok=True)

    pc_sizes = [5_000_000, 10_000_000, 20_000_000]

    combinations = list(product(stl_files, pc_sizes))

    with Pool() as pool:
        pool.starmap(gen_pc, combinations)
