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
  - An inference recipe calling a pre-trained AI models that were trained using
PhysicsNeMo and hosted as NVIDIA Inference Microservices (for example, the
[DoMINO Automotive Aerodynamics
NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/domino-automotive-aero))
from a Python interface, facilitating scalable deployment of trained models.
- **Benchmarking of ML Model Accuracy**:
  - A benchmark for evaluating and validating the results of trained ML models
  against traditional CFD results using broad set of built-in engineering
  metrics (for example, pointwise errors, integrated quantities, spectral
  metrics, PDE residuals). [Related publication](https://www.arxiv.org/abs/2507.10747)
  - Utilities to extend and build custom metrics, analyze, and visulaize the
    results of trained ML model, both mesh-based and point-cloud based models

- **Hybrid Initialization**:
  - An end-to-end recipe illustrating initializing a CFD simulation with a
  trained ML model hybridized with potential flow solutions, to accelerate CFD
  convergence (particularly for high-fidelity, unsteady cases). [Related
  publication](https://arxiv.org/abs/2503.15766)

## Installation

PhysicsNeMo-CFD is a Python package that depends on the [NVIDIA PhysicsNeMo
framework](https://github.com/NVIDIA/physicsnemo).

PhysicsNeMo-CFD depends on PhysicsNeMo. PhysicsNeMo is a dependency
of PhysicsNeMo-CFD and the below pip installation command will install
PhysicsNeMo automatically if not present.

For maximum cross-platform compatibility, we recommend using the PhysicsNeMo
Docker container. Steps to use [PhysicsNeMo container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/containers/physicsnemo)
can be found [in the Getting Started](https://docs.nvidia.com/deeplearning/physicsnemo/getting-started/index.html#physicsnemo-with-docker-image-recommended).

You can install PhysicsNeMo-CFD via pip:

```bash
git clone https://github.com/NVIDIA/physicsnemo-cfd.git
cd physicsnemo-cfd
pip install .
```

To get access to GPU accelerated functionalities from this repo when installing
in a conda or a custom python environment please run below commands.

If you are using the [PhysicsNeMo container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/containers/physicsnemo),
the GPU specific dependencies are pre-installed, hence this additional step is
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

To get started use the DoMINO NIM on a sample as shown below:

```python
from physicsnemo.cfd.inference.domino_nim import call_domino_nim
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

Refer to the [`workflows` directory](./workflows) for detailed instructions on
executing individual reference workflows and samples. These are primarily
packaged as Jupyter notebooks where possible, to provide for inline
documentation and visualization of expected results.

## Contributing to PhysicsNeMo

PhysicsNeMo is an open source collaboration and its success is rooted in
community contribution to further the field of Physics-ML. Thank you for
contributing to the project so others can build on top of your contribution.

For guidance on contributing to PhysicsNeMo, refer to the [contributing
guidelines](CONTRIBUTING.md).

## Cite PhysicsNeMo

If PhysicsNeMo helped your research and you would like to cite it, refer to the
[guidelines](https://github.com/NVIDIA/physicsnemo/blob/main/CITATION.cff).

## Communication

- Github Discussions: Discuss new architectures, implementations, and Physics-ML
  research.
- GitHub Issues: Bug reports, feature requests, and install issues.
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
