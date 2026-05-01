# PhysicsNeMo CFD
<!-- markdownlint-disable -->

[![Project Status: Active - The project has reached a stable, usable state and
is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/physicsnemo)](https://github.com/NVIDIA/physicsnemo/blob/master/LICENSE.txt)
[![Code style:
black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- markdownlint-enable -->
 [**PhysicsNeMo CFD**](#what-is-physicsnemo-cfd) | [**Getting
started**](#getting-started) | [**Contributing
Guidelines**](#contributing-to-physicsnemo) |
[**Communication**](#communication)

## What is PhysicsNeMo CFD?

NVIDIA PhysicsNeMo-CFD is a sub-module of [NVIDIA PhysicsNeMo
framework](https://github.com/NVIDIA/physicsnemo/) that provides the tools
needed to integrate pretrained AI models into engineering and CFD workflows.

The library is a collection of loosely-coupled workflows around the trained AI
models for CFD, with abstractions and relevant data structures.

Refer to the [PhysicsNeMo
framework](https://github.com/NVIDIA/physicsnemo/blob/main/README.md) to learn
more about the full stack.

The library offers utilities for:

- **NIM Inference**:
  - An inference recipe calling pre-trained AI models that were trained using
PhysicsNeMo and hosted as NVIDIA Inference Microservices (for example, the
[DoMINO Automotive Aerodynamics
NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/domino-automotive-aero))
from a Python interface, facilitating scalable deployment of trained models.
- **Benchmarking of ML Model Accuracy**:
  - A benchmark for evaluating and validating the results of trained ML models
  against traditional CFD results using a broad set of built-in engineering
  metrics (for example, pointwise errors, integrated quantities, spectral
  metrics, PDE residuals). [Related publication](https://www.arxiv.org/abs/2507.10747)
  - The `physicsnemo.cfd.evaluation` package runs config-driven inference and
    uses the same `physicsnemo.cfd.postprocessing_tools` metric implementations as the
    **[`workflows/benchmarking/`](workflows/benchmarking/)** Hydra workflow (run **`python main.py`** from that directory; see **[that README](workflows/benchmarking/README.md)**).
    **Pretrained checkpoints** for built-in benchmark models are published on **[Hugging Face](https://huggingface.co/nvidia)** under the `nvidia/*_drivaerml` model repositories (pinned package roots in **[`builtin_packages.py`](physicsnemo/cfd/evaluation/assets/builtin_packages.py)**).
    The benchmark **evaluation dataset** is **[DrivAerML](https://huggingface.co/datasets/neashton/drivaerml)** on Hugging Face. For air-gapped or custom layouts, set local checkpoint and dataset paths in YAML or Hydra overrides.
  - **Custom models, data, and metrics:** you can plug in additional **`CFDModel`** wrappers, **`DatasetAdapter`** implementations, **metrics**, and optional **report visuals** for the same Hydra benchmark harness (no fork required). See **[Custom models, datasets, and metrics](workflows/benchmarking/README.md#custom-models-datasets-and-metrics)** in the benchmarking workflow README and the **`physicsnemo.cfd.evaluation`** registration APIs (`register_metric`, `register_visual`, model registry).
  - Utilities to analyze and visualize predictions from trained ML models (mesh-based and point-cloud), including outputs produced with custom metrics.

- **Hybrid Initialization**:
  - An end-to-end recipe for initializing a CFD simulation with a
  trained ML model hybridized with potential flow solutions, to accelerate CFD
  convergence (particularly for high-fidelity, unsteady cases). [Related
  publication](https://arxiv.org/abs/2503.15766)

## Installation

PhysicsNeMo-CFD is a Python package that depends on the [NVIDIA PhysicsNeMo
framework](https://github.com/NVIDIA/physicsnemo).

PhysicsNeMo-CFD depends on PhysicsNeMo. The pip installation command below will install
PhysicsNeMo automatically if not present.

For maximum cross-platform compatibility, we recommend using the PhysicsNeMo
Docker container. Steps to use the [PhysicsNeMo container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/containers/physicsnemo)
can be found in the [Getting Started guide](https://docs.nvidia.com/deeplearning/physicsnemo/getting-started/index.html#physicsnemo-with-docker-image-recommended).

You can install PhysicsNeMo-CFD via pip:

```bash
git clone https://github.com/NVIDIA/physicsnemo-cfd.git
cd physicsnemo-cfd
pip install .
```

For **local development** (editable install, tests, and benchmarking workflow checks), use optional **dev** dependencies from [`pyproject.toml`](pyproject.toml):

```bash
pip install -e ".[dev]"
```

To get access to GPU-accelerated functionalities from this repository when installing
in a conda or custom Python environment, please run the commands below.

If you are using the [PhysicsNeMo container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/containers/physicsnemo),
the GPU-specific dependencies are pre-installed, so this additional step is
not required.

```bash
pip install .[gpu] --extra-index-url=https://pypi.nvidia.com
```

> [!Note] PhysicsNeMo-CFD is an experimental library and currently v0; expect
> breaking changes. PhysicsNeMo-CFD is for *demonstrating* workflows, rather
than providing a stable API for production-level deployments.

When updating, see the latest changes in the [CHANGELOG.md](./CHANGELOG.md)
file.

## Getting started

To get started, use the DoMINO NIM on a sample as shown below:

```python
from physicsnemo.cfd.evaluation.nims import call_domino_nim
import subprocess

filenames = [
    "drivaer_202.stl",
]
urls = [
    "https://huggingface.co/datasets/neashton/drivaerml/resolve/main/run_202/drivaer_202.stl",
]

for url, filename in zip(urls, filenames):
    subprocess.run(["wget", url, "-O", filename], check=True)

output_dict = call_domino_nim(
    stl_path="./drivaer_202.stl",
    inference_api_url="http://localhost:8000/v1/infer",
    data={
        "stream_velocity": "38.89",
        "stencil_size": "1",
        "point_cloud_size": "500000",
    },
    verbose=True,
)

```

Reference workflows live under the [`workflows`](./workflows) directory. The
**[benchmarking workflow](workflows/benchmarking/)** is the supported path for
config-driven model evaluation and metrics (`python main.py` with Hydra). Older
file-based benchmarking samples were moved to
**[workflows/deprecated/bench_example](workflows/deprecated/bench_example/)** and are
superseded by `benchmarking`. Other samples may be packaged as Jupyter
notebooks for inline documentation and visualization.

## Contributing to PhysicsNeMo

PhysicsNeMo is an open-source collaboration and its success is rooted in
community contributions to further the field of Physics-ML. Thank you for
contributing to the project so others can build on top of your contributions.

For guidance on contributing to PhysicsNeMo, refer to the [contributing
guidelines](CONTRIBUTING.md). Changes to **`physicsnemo.cfd.evaluation`** or the
**benchmarking workflow** should follow the tests and notes in **CONTRIBUTING.md**
(including the recommended `pytest` command for `test/ci_tests/`).

## Cite PhysicsNeMo

If PhysicsNeMo helped your research and you would like to cite it, refer to the
[guidelines](https://github.com/NVIDIA/physicsnemo/blob/main/CITATION.cff).

## Communication

- GitHub Discussions: Discuss new architectures, implementations, and Physics-ML
  research.
- GitHub Issues: Bug reports, feature requests, and installation issues.
- PhysicsNeMo Forum: The [PhysicsNeMo
Forum](https://forums.developer.nvidia.com/t/welcome-to-the-physicsnemo-ml-model-framework-forum/178556)
hosts an audience of new to moderate-level users and developers for general
chat, online discussions, and collaboration.

## Feedback

Want to suggest some improvements to PhysicsNeMo? Use our [feedback
form](https://docs.google.com/forms/d/e/1FAIpQLSfX4zZ0Lp7MMxzi3xqvzX4IQDdWbkNh5H_a_clzIhclE2oSBQ/viewform?usp=sf_link).

## License

PhysicsNeMo is provided under the Apache License 2.0, see
[LICENSE.txt](./LICENSE.txt) for full license text.
