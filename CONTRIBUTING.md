# PhysicsNeMo Contribution Guide

## Introduction

Welcome to Project PhysicsNeMo! We're excited you're here and want to contribute.
This documentation is intended for individuals and institutions interested in
contributing to PhysicsNeMo. PhysicsNeMo is an open-source project and, as such,
its success relies on its community of contributors willing to keep improving it.
Your contribution will be a valued addition to the codebase; we simply ask
that you read this page and understand our contribution process, whether you
are a seasoned open-source contributor or a first-time
contributor.

### Communicate with Us

We are happy to talk with you about your needs for PhysicsNeMo and your ideas
for contributing to the project. One way to do this is to create an issue
discussing your thoughts. It might be that a very similar feature is under
development or already exists, so an issue is a great starting point. If you are
looking for an issue to resolve that will help, refer to the
[issue](https://github.com/NVIDIA/physicsnemo/issues) section. If you are
considering collaborating with the NVIDIA PhysicsNeMo team to enhance
PhysicsNeMo, fill out this [proposal form](https://forms.gle/fYsbZEtgRWJUQ3oQ9)
and we will get back to you.

## Contribute to PhysicsNeMo-CFD

### Pull Requests

The developer workflow for code contributions is as follows:

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo)
the [upstream](https://github.com/NVIDIA/physicsnemo-cfd) PhysicsNeMo-CFD repository.

2. Clone the forked repository and push changes to the personal fork.

3. Once the code changes are staged on the fork and ready for review, a
[Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR)
can be [requested](https://help.github.com/en/articles/creating-a-pull-request)
to merge the changes from a branch of the fork into a selected branch of upstream.

    - Exercise caution when selecting the source and target branches for the PR.
    - Ensure that you update the [`CHANGELOG.md`](CHANGELOG.md) to reflect your contributions.
    - Creating a PR kicks off CI and a code review process.
    - At least one PhysicsNeMo engineer will be assigned for the review.

4. The PR will be accepted and the corresponding issue closed after adequate
review and testing have been completed. Note that every PR should correspond to
an open issue and should be linked on GitHub.

### Licensing Information

All source code files should start with this paragraph:

```bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
```

### Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies
that the contribution is your original work, or you have rights to submit it
under the same license, or a compatible license.

  - Any contribution that contains commits that are not signed off will not be accepted.

- To sign off on a commit, simply use the `--signoff` (or `-s`) option when
committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```text
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license 
    document, but changing it is not allowed.
  ```

  ```text
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to 
    submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge,
    is covered under an appropriate open source license and I have the right under that
    license to submit that work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am permitted to submit under a
    different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified
    (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and
    that a record of the contribution (including all personal information I submit with
    it, including my sign-off) is maintained indefinitely and may be redistributed
    consistent with this project or the open source license(s) involved.

  ```

### Pre-commit

For PhysicsNeMo development, [pre-commit](https://pre-commit.com/) is **required**.
This will not only help developers pass the CI pipeline but also accelerate reviews.
Contributions that have not used pre-commit will *not be reviewed*.

To install `pre-commit`, follow the steps below inside the PhysicsNeMo repository
folder:

```bash
pip install pre-commit
pre-commit install
```

Once the above commands are executed, the pre-commit hooks will be activated and
all commits will be checked for appropriate formatting.

### Developing and testing the benchmarking workflow

Install the package in editable mode with dev dependencies from the repository root
([`pyproject.toml`](pyproject.toml)):

```bash
pip install -e ".[dev]"
```

For Hugging Face / fsspec-backed **`model.package`** URIs in benchmarks, also install
**`pip install -e ".[evaluation-hf]"`** (optional extra in [`pyproject.toml`](pyproject.toml)).

### Model checkpoints and remote packages (`evaluation.assets`)

Built-in and custom **`CFDModel`** wrappers load weights via
**`load(checkpoint_path, stats_path, …)`**. The benchmark engine resolves paths
first using **`physicsnemo.cfd.evaluation.assets.resolve_model_assets`**:

| Tier | Use case | Configuration |
| ---- | -------- | --------------- |
| **A** | Private or local weights | Set both **`model.checkpoint`** and **`model.stats_path`** (no `package`). |
| **B** | Weights in a Hub/S3/local tree | Set **`model.package`** (e.g. `hf://org/repo@revision`) and **`checkpoint_relpath`** / **`stats_relpath`**, or the same keys under **`model.kwargs`**. |
| **C** | Default URI for a registered name | First-party models use **`physicsnemo.cfd.evaluation.assets.builtin_packages`** (per-model **`GEOTRANSOLVER_PACKAGE_ROOT`**, etc., and **`register_builtin_model_packages`**). Third parties call **`register_default_asset(name, AssetSpec(...))`**; then **`checkpoint`** / **`stats_path`** may be omitted if **`REQUIRES_REMOTE_ASSETS`** is true. |

Stubs that do not load weights (e.g. **`surface_baseline`**,
**`volume_baseline`**) set **`REQUIRES_REMOTE_ASSETS = False`** on the wrapper
class so empty checkpoint/stats remain valid.

Optional Hub analytics: the engine performs a best-effort resolve of
**`config.json`** inside HF-backed packages (ignored if missing). **`ngc://`**
roots are not implemented yet (constructor/`resolve` will raise
**`NotImplementedError`**).

Before opening a pull request that changes **`physicsnemo/cfd/evaluation/`** or
**`workflows/benchmarking/`**, run **`test/ci_tests/`** (everything under that folder):

```bash
pytest -q test/ci_tests/
```

The **`compute_error_vs_sdf`** tests in **`test_l2_errors.py`** skip cleanly
when **`warp`** is unavailable (install via the **`gpu`** optional extra in
[`pyproject.toml`](pyproject.toml)).

Do **not** commit generated benchmark outputs under
**`workflows/benchmarking/`** (e.g. `benchmark_results*/`, `gpu_output.log`,
`metrics_cache/`, Hydra outputs under the run directory). That directory
includes a **`.gitignore`** for common artifacts.

**Exit codes:** **`workflows/benchmarking/main.py`** calls
**`run_benchmark_cli`**, which maps **`BenchmarkPolicyError`** (e.g. when
**`run.fail_on_all_skipped`** or **`run.fail_on_any_metric_nan`** triggers) to
exit **1**. See **`workflows/benchmarking/README.md`** for details.

**Continuous integration:**
[`.github/workflows/ci-tests.yml`](.github/workflows/ci-tests.yml) and
[`.gitlab-ci.yml`](.gitlab-ci.yml) run the same **`pytest`** path on push/pull
request (or merge request to **`main`**). Additional hosts should use
**`pytest`** on **`test/ci_tests/`** so regressions cannot slip past one CI but
not another.

Update [`CHANGELOG.md`](CHANGELOG.md) for user-visible changes to evaluation or
benchmark behavior and configuration.
