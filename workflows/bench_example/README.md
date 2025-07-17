# Benchmarking AI surrogates for External Aerodynamics

The benchmarking workflow is designed for evaluating and validating AI models
for external aerodynamics. Predicting accurate aerodynamic characteristics of a
vehicle (car / aircraft) requires analysis of both surface and volume fields.
Surface predictions are useful for computing quantities such as drag and lift,
essential for evaluating the performance and efficiency of the vehicle design.
The volumetric predictions are essential for analyzing the flow field around the
vehicle, like vortices and wake structures.
Refer to this [related publication](https://www.arxiv.org/abs/2507.10747) for more
details.

## Using the benchmarking workflows

The built-in benchmarking code is organized as two scripts,
[`generate_surface_benchmarks.py`](./generate_surface_benchmarks.py) and
[`generate_volume_benchmarks.py`](./generate_volume_benchmarks.py). We also
provide [notebook](./notebooks/) versions of these that can be used to deepen
the understanding of several metrics and doing the comparisons with more
flexibility and freedom.

To keep the handling of various models simple, these workflows take AI model
predictions post-processed to `.vtp` (for surface) and `.vtu` (for volume)
formats as inputs. Generating these `vtp`, `vtu` files depends on the model's
architecture and design, and hence we leave that responsibility to the user. The
[notebooks](./notebooks/) provide steps to demonstrate the inference using the
[DoMINO Automotive-Aero
NIM](https://docs.nvidia.com/nim/physicsnemo/domino-automotive-aero/latest/overview.html)
which can be used as a reference.

Typically, one can simply write the predicted results to the original `.vtu` or
`.vtp` files by following some simple steps as shown below.

```python
import pyvista as pv
import numpy as np

true_data = pv.read("./boundary_100.vtp")   # Reading a sample boundary file from the DrivAerML dataset
true_data.point_data["pMeanTrimPred"] = np.random.rand(*true_data.point_data["pMeanTrim"])  # Sample model's predicitons

true_data.save("./boundary_100_with_model_predictions.vtp")
```

> **Note**: The benchmarking workflows only work with post-processed
`.vtp` and `.vtu` files. Inferencing workflows other than the DoMINO NIM
are presently out of the scope. For models present in PhysicsNeMo, please
refer to the [`examples/cfd/external_aerodynamics`](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics)
directory of PhysicsNeMo for training and testing / inference code.

### Executing the workflows

The workflows are designed to be run via command line and can be configured by
passing additional commandline arguments.

```bash
usage: generate_surface_benchmarks.py [-h] [--pc-results-dir PC_RESULTS_DIR] [-n NUM_PROCS]
                                      [--output-dir OUTPUT_DIR]
                                      [--contour-plot-ids CONTOUR_PLOT_IDS [CONTOUR_PLOT_IDS ...]]
                                      [--field-mapping FIELD_MAPPING]
                                      sim_results_dir

usage: generate_volume_benchmarks.py [-h] [-n NUM_PROCS] [--output-dir OUTPUT_DIR]
                                     [--contour-plot-ids CONTOUR_PLOT_IDS [CONTOUR_PLOT_IDS ...]]
                                     [--field-mapping FIELD_MAPPING]
                                     sim_results_dir
```

> **Note**: The benchmarking workflows and notebooks require `xvfb` installed.
To install it, please run `sudo apt-get install xvfb`, or run `bash setup.sh`

#### Sample commands to run the benchmarking workflows

Let's assume the data is stored as below:

```bash
tree sample_data/
sample_data/
├── surface_data
│   ├── predicted_surface_run1_reference.vtp
│   └── predicted_surface_run2_reference.vtp
└── volume_data
    ├── predicted_volume_run1_reference.vtu
    └── predicted_volume_run2_reference.vtu
```

##### Surface benchmarking

The surface files have below variables:

```python
>>> import pyvista as pv
>>> pv.read("./sample_data/surface_data/predicted_surface_run1_reference.vtp").cell_data.keys()
['U', 'UMean', 'k', 'kMean', 'nut', 'nutMean', 'omega', 'omegaMean', 'p', 'pMean', 'wallShearStress', 'wallShearStressMean', 'pMeanPred', 'wallShearStressMeanPred']
```

To run the surface benchmarking workflow on these files, we can run the
following command:

```bash
python generate_surface_benchmarks.py \
    ./sample_data/surface_data \
    --num-procs 2 \
    --output-dir ./sample_surface_plots \
    --field-mapping '{"p":"pMean", "wallShearStress":"wallShearStressMean", "pPred":"pMeanPred", "wallShearStressPred":"wallShearStressMeanPred"}' \
    --contour-plot-ids 1 2
```

This will generate an output similar to below:

```bash
Processing: ./sample_data/surface_data/predicted_surface_run1_reference.vtp, None
Processing: ./sample_data/surface_data/predicted_surface_run2_reference.vtp, None
L2 Errors for pMean_l2_error: 0.20886313915252686
L2 Errors for wallShearStressMean_x_l2_error: 0.2778988182544708
L2 Errors for wallShearStressMean_y_l2_error: 0.5308563709259033
L2 Errors for wallShearStressMean_z_l2_error: 0.3680238127708435
Area weighted L2 Errors for pMean_area_wt_l2_error: 0.17918650478125087
Area weighted L2 Errors for wallShearStressMean_x_area_wt_l2_error: 0.22027747158422645
Area weighted L2 Errors for wallShearStressMean_y_area_wt_l2_error: 0.4785210617133781
Area weighted L2 Errors for wallShearStressMean_z_area_wt_l2_error: 0.34711867679980934
Plotting contour plots for ['1', '2']
```

###### Evaluations on Point Clouds

The surface benchmarking workflows support predictions on point clouds. These
are added as optional since in certain scenarios, the requirement to have a
simulation mesh to infer the model results can reduce its practical use. Meshing
is a fairly time consuming process in the worflow due to it being difficult to
parallelize. Sampling pointclounds on the other hand does not face these
challenges and can be computed in fraction of seconds making it ideal for
real-time inferencing workflows.

We provide a sample script that computes such point clounds on the surface. We
use PhysicsNeMo-Sym library's point cloud sampling functionalities to achive
this. Code to generate the evaluation point clounds is provided in
[`generate_pcs_from_stl.py`](./generate_pcs_from_stl.py).

Note, a similar script can be written for the volume point clouds as well (using
the `sample_interior` function PhysicsNeMo-Sym)

##### Volume benchmarking

The volume files have below variables:

```python
>>> import pyvista as pv
>>> pv.read("./sample_data/volume_data/predicted_volume_run1_reference.vtu").point_data.keys()
['k', 'kMean', 'nut', 'nutMean', 'omega', 'omegaMean', 'p', 'pMean', 'U', 'UMean', 'wallShearStress', 'wallShearStressMean', 'UMeanPred', 'pMeanPred', 'kMeanPred', 'nutMeanPred']
```

To run the volume benchmarking workflow on these files, we can run the following
command:

```bash
python generate_volume_benchmarks.py \
    ./sample_data/volume_data \
    --num-procs 2 \
    --output-dir ./sample_volume_plots \
    --field-mapping '{"p":"pMean", "U":"UMean", "nut":"nutMean", "pPred":"pMeanPred", "UPred":"UMeanPred", "nutPred":"nutMeanPred"}' \
    --contour-plot-ids 1 2
```

This will generate an output similar to below:

```bash
Processing: ./sample_data/volume_data/predicted_volume_run2_reference.vtu
Processing: ./sample_data/volume_data/predicted_volume_run1_reference.vtu
L2 Errors for pMean_l2_error: 0.21289117634296417
L2 Errors for UMean_x_l2_error: 0.11827605217695236
L2 Errors for UMean_y_l2_error: 0.3803640902042389
L2 Errors for UMean_z_l2_error: 0.3300434947013855
L2 Errors for nutMean_l2_error: 0.23974356055259705
L2 Errors for Continuity_l2_error: 155.8505539704951
Plotting contour plots for ['1', '2']

```

## Using standardized datasets for inter-model comparisons

To make the comparisons of different models fair and consistent, it is important
to train and evaluate the models on a reference dataset. Here, we demonstrate
the use of the DrivAerML dataset for benchmarking the ML models. The DrivAerML
dataset is a high-fidelity, open-source dataset designed to advance machine
learning applications in automotive aerodynamics. It includes 500
parameterically varied geometries based on the widely used DrivAer notchback
vehicle model, generated using hybrid RANS-LES (HRLES) simulations.

To keep the comparisons consistent, we encourage users to use the
train-validation split as suggested in
[./drivaer_ml_files/train.csv](./drivaer_ml_files/train.csv) and
[./drivaer_ml_files/validation.csv](./drivaer_ml_files/validation.csv). Please
refer to [./drivaer_ml_files/README.md](./drivaer_ml_files/README.md) for more
details.

The results of the DoMINO, XAeroNet, and FigNet models from
[`PhysicsNeMo`](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics)
evaluated on the DrivAerML dataset with the above train-validation split, and
using the [`generate_surface_benchmarks.py`](./generate_surface_benchmarks.py)
workflow are tabulated below.

### Surface benchmarking results

Results from surface benchmarking on the validation set:

#### L2 Errors (Surface)

|                         | XAeroNet | FigNet | DoMINO |
|-------------------------|----------|--------|--------|
| Pressure                | 0.1401   | 0.2071 | 0.0969 |
| Wall Shear   Stress (x) | 0.1741   | 0.3184 | 0.1823 |
| Wall Shear   Stress (y) | 0.2238   | 0.6227 | 0.2601 |
| Wall Shear   Stress (z) | 0.2863   | 0.5336 | 0.2786 |

#### Area weighted L2 Errors (Surface)

|                         | XAeroNet | FigNet | DoMINO |
|-------------------------|----------|--------|--------|
| Pressure                | 0.1445   | 0.1402 | 0.0755 |
| Wall Shear   Stress (x) | 0.1337   | 0.1574 | 0.1015 |
| Wall Shear   Stress (y) | 0.2450   | 0.3417 | 0.2308 |
| Wall Shear   Stress (z) | 0.2463   | 0.2945 | 0.1852 |

#### Cd R2 Scores

|    | XAeroNet | FigNet | DoMINO |
|----|----------|--------|--------|
| R2 | 0.9206   | 0.9749 | 0.9834 |

#### Cd Design Trend analysis

|                       | XAeroNet | FigNet  | DoMINO  |
|-----------------------|----------|---------|---------|
| Spearman Coeff        | 0.9600   | 0.9870  | 0.9940  |
| Max Abs. Error   (N)  | 58.2000  | 25.7000 | 23.1000 |
| Mean Abs.   Error (N) | 15.2000  | 8.8600  | 6.6400  |

### Volume benchmarking results

Results from volume benchmarking on the validation set:

#### L2 Errors (Volume)

|                     | DoMINO |
|---------------------|--------|
| Pressure            | 0.1042 |
| Velocity (x)        | 0.0948 |
| Velocity (y)        | 0.1848 |
| Velocity (z)        | 0.2040 |
| Turbulent Viscosity | 0.3404 |
