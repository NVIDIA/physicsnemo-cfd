# Hybrid Initialization Reference Workflow for Transient Automotive Aerodynamics

This reference workflow demonstrates NVIDIA's hybrid initialization approach for
accelerating transient automotive aerodynamics simulations using OpenFOAM-based
solvers. The workflow combines a potential flow solution with a machine learning
surrogate initialization from the DoMINO NVIDIA Inference Microservice (NIM).

## Overview

The workflow architecture is shown below. This repository contains the
components in the "OpenFOAM + Python Docker Container" box, along with
instructions for launching the DoMINO NIM Docker Container and an example
OpenFOAM-based case:

![workflow](./assets/workflow.drawio.png)

This hybrid initialization workflow consists of six main stages:

1. **DoMINO NIM Setup**: Launch the DoMINO Automotive Aero NIM container that
   provides ML-based flow field predictions
2. **Asset Preparation**: Download vehicle geometry and generate ML flow field
   predictions using DoMINO NIM
3. **Mesh Generation**: Create a high-quality computational mesh around the
   vehicle geometry
4. **Potential Flow Solution**: Compute a baseline potential flow solution for
   the case
5. **Hybrid Initialization**: Combine the potential flow and ML predictions to
   create optimized initial conditions
6. **Transient CFD Simulation**: Run the full transient CFD simulation with the
   hybrid initialization

The key innovation is step 5, where machine learning predictions from DoMINO are
intelligently blended with physics-based potential flow solutions to provide
superior initial conditions that accelerate subsequent transient convergence.

## Prerequisites

- Docker and NVIDIA Container Toolkit installed
- NVIDIA NGC API key (for downloading DoMINO NIM checkpoints)
- Sufficient disk space (~50GB for the baseline case)
- At least 64GB RAM recommended

## Running the Workflow

### Step 1: Launch the DoMINO NIM Container

First, you'll need to launch the DoMINO NIM container that provides ML-based
flow predictions. There are two options to achieve this; in either case, you
will need a NVIDIA NGC API key to download the latest model checkpoint files. If
you don't already have one, you can obtain one
[here](https://org.ngc.nvidia.com/setup/api-keys) after registering for a NVIDIA
NGC account.

**Option A - Using the provided script (recommended):**

```bash
# Navigate to the repository root directory
cd /path/to/physicsnemo-cfd

# Set your NGC API key
export NGC_API_KEY="your_ngc_api_key_here"
# Tip: for a more long-term solution, you can set this in your shell profile.

# Launch DoMINO NIM (Note: this requires an internet connection to download model checkpoints)
./physicsnemo/cfd/inference/launch_local_domino_nim.sh
```

**Option B - Manual setup:**

Follow the detailed instructions
[here](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/domino-automotive-aero).
This allows more fine-grained control over the container launch process.

### Step 2: Launch the OpenFOAM Container Environment

In a separate shell, attempt to launch the OpenFOAM container:

```bash
# Navigate to the repository root (if not already there)
cd /path/to/physicsnemo-cfd

# Launch the OpenFOAM + Python container with proper mounting
# Note: for proper path mounting, this script MUST be run as-shown from the repository root directory.
./workflows/hybrid_initialization_example/openfoam_interactive.sh
```

If this command is successful, you should be inside the container, with the
entire `physicsnemo-cfd` repository mounted to `/workspace/`:

```bash
root@8a663debb154:/workspace# ls
CHANGELOG.md  CONTRIBUTING.md  LICENSE.txt  Makefile  README.md  SECURITY.md  assets  build  nvidia_physicsnemo_cfd.egg-info  physicsnemo  pyproject.toml  test  workflows
```

Alternatively, you may encounter an error indicating that the container image is
not found. If so, you will be prompted to build the container locally before
launching:

```bash
Docker image 'openfoam-python:latest' not found locally.
Would you like to build it now? (this may take several hours) [y/N]: 
```

You can also manually trigger a container build by running `make container` in
the [`container/`](./container/) directory.

> **⏰ Expected Build Time**: Building the container may take 3+ hours to
> complete, depending on hardware and network speed, and whether any
> sub-components of the container have already been built.

Note that this image only contains OpenFOAM and Python, plus the underlying
dependencies for `physicsnemo-cfd`. It does not contain `physicsnemo-cfd`
pre-installed, which is intended to reduce the need to re-build the container
when `physicsnemo-cfd` is updated. It also does not contain the DoMINO
Automotive Aero NIM, which is [available
separately](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/domino-automotive-aero).

> **Alternative for advanced users:** If you prefer not to use the OpenFOAM
> container, you can install required dependencies locally instead, and skip
> container usage. These dependencies are Python 3.10+ and OpenFOAM v2206;
> ensure that respective binaries are available on `PATH`. If you encounter
> dependency issues based on this approach, we recommend using the standard
> container approach instead. Note that the following steps assume that you are
> inside the container, so syntax may need to be slightly modified if this is
> not the case.

### Step 3: Install PhysicsNeMo-CFD (Inside Container)

Once inside the container, install the package and its dependencies:

```bash
# You should now be in the container at /workspace/ 
pip install . --extra-index-url=https://pypi.nvidia.com
```

### Step 4: Run the Complete Workflow (Inside Container)

Execute the main workflow script:

```bash
# Navigate to the workflow directory
cd workflows/hybrid_initialization_example/

# Run the workflow
./run.sh
```

> **⏰ Expected Runtime**: This workflow may require 7+ hours to complete on
> typical hardware. Most of the time is spent on the transient CFD simulation.
> For testing purposes only, the transient CFD simulation can be interrupted
> after a few time steps with CTRL+C, and the script will continue to
> post-processing.

The script will automatically perform all workflow stages:

1. Prepare DoMINO assets (geometry and predicted flow field)
2. Generate the computational mesh
3. Compute the potential flow solution  
4. Create the hybrid initialization
5. Run the transient CFD simulation
6. Generate VTK output for visualization

### Step 5: View Results

After completion, results will be available in:

- `workflows/hybrid_initialization_example/VTK/` directory: Visualization files
- `workflows/hybrid_initialization_example/postProcessing/` directory: Force
  time-histories and other analysis data

## Troubleshooting and Important Notes

**Directory mounting issues**: Always run `openfoam_interactive.sh` from the
repository root directory to ensure proper file mounting

**Workflow errors**: Check the various `log.*` files produced at each step to
diagnose issues. The `run.sh` script continues on errors, so look for the first
error in the logs

**Reset workflow**: Run `./clean.sh` to reset the directory to its original
state

## Expected Results

### Mesh Generation

The workflow generates a high-quality mesh suitable for automotive aerodynamics:

![mesh](./assets/mesh_lq.jpg)

### Hybrid Initialization

The initialization field combines potential flow and ML predictions:

![hybrid_U](./assets/hybrid_U.jpg)

### Final Flow Field

The transient simulation produces a detailed flow field:

![flow](./assets/total_pressure_isosurface.jpg)

### Performance Metrics

The hybrid initialization significantly accelerates convergence compared to
traditional methods:

![force_plot_raw](./assets/force_plot_raw.png)

The workflow in this folder (without modification) should roughly replicate the
line labeled "DoMINO + Potential (k-based hybrid)" in the chart above. Results
will not be identical, due to the chaotic nature of turbulent structures in the
vehicle wake, but the general trend should be similar.

For detailed performance analysis and methodology, please refer to our
publication:

- Peter Sharpe, Rishikesh Ranade, Kaustubh Tangsali, Mohammad Amin Nabian, Ram
  Cherukuri, Sanjay Choudhry, ["Accelerating Transient CFD through Machine
  Learning-Based Flow Initialization"](https://arxiv.org/abs/2503.15766), 2025.

## Customizing the Workflow

### Modifying the Workflow for Different Physical Cases

The [`run.sh`](run.sh) script begins with a Python script,
`prepare_domino_assets.py`, which generates intermediate files in the
[`from_domino/`](./from_domino/) directory:

- `initialConditions`, a C++-like header file that contains constants for the
  inlet velocity, outlet pressure, and turbulence parameters. A template for
  this file format is included here; this could be modified for different
  OpenFOAM cases if desired, though the template includes important parameters
  for many external flow cases.
- `vehicle.stl` (140 MB for the baseline DrivAerML ID 4 case), the geometry of
  the vehicle. This is downloaded from a public HuggingFace repository.
- `predicted_flow.vtu` (38 GB for the baseline DrivAerML ID 4 case), which
  represents the DoMINO-predicted flow field. This is generated on-the-fly using
  a call to the DoMINO NIM. If you wish to directly modify the predicted flow
  field, note the format - this should be a VTK unstructured grid file with (at
  least) the following point data fields:
  - `UMeanTrimPred`, the predicted time-averaged velocity field.
  - `pMeanTrimPred`, the predicted time-averaged pressure field.
  - `TKEPred`, the predicted turbulent kinetic energy (k) field.
  - `nutMeanTrimPred`, the predicted turbulent viscosity field.

By modifying these inputs (either directly, or by modifying the
`prepare_domino_assets.py` script), the workflow can be adapted to different
cases.

### Modifying the Workflow for Different CFD Solvers

To run this workflow with a different CFD solver for the transient solve, two
modifications should be made:

- First, modify the
  [`make_hybrid_initialization.py`](make_hybrid_initialization.py) script to
  generate the potential flow solution using the new solver. Interfaces are
  provided for either VTU or OpenFOAM polyMesh directories as input; other
  formats may require third-party conversion tools. You will also need to modify
  the hybrid initialization write step to match the new solver's expected format
  for field data.
- Second, modify the [`run.sh`](run.sh) script to use the new solver for the
  transient CFD solve.
