PhysicsNeMo CFD Model Inference
===============================

.. automodule:: physicsnemo.cfd.evaluation.inference
   :no-members:
   :no-inherited-members:

.. currentmodule:: physicsnemo.cfd.evaluation.inference

.. autosummary::
   :nosignatures:
   :toctree: generated/inference/

   CFDModel
   OutputLocation
   register_model
   get_model_wrapper
   get_output_location_for_model
   get_inference_domain_for_model
   list_models

Model Wrappers
--------------

.. currentmodule:: physicsnemo.cfd.evaluation.inference.wrappers

.. autosummary::
   :nosignatures:
   :toctree: generated/inference/

   DominoWrapper
   FIGNetWrapper
   XMGNWrapper
   GeoTransolverWrapper
   TransolverWrapper
   SurfaceBaselineWrapper
   VolumeBaselineWrapper

Shared UQ statistics (opt-in)
----------------------------

Set ``run.uq.statistics_backend: physicsnemo`` to use PhysicsNeMo's tensor-mode
``Mean`` and ``Variance`` accumulators for sampling inference. The default
``numpy`` backend retains the existing behavior. This option requires a
PhysicsNeMo version providing ``Variance(mode="tensor")``, introduced in
`PhysicsNeMo PR #2188 <https://github.com/NVIDIA/physicsnemo/pull/2188>`_.
PhysicsNeMo 2.2.0 supports the default NumPy backend; it does not provide this
opt-in API. Selecting the new backend on an older version raises an explanatory
error before sampling.

Both paths decode each prediction before accumulating it and retain float64
statistics in physical units. Population variance across member means is added
to the mean within-member variance for GP ensembles. Prediction containers,
retained samples, and metric interpretation remain the same. The option enters
the metrics-cache fingerprint.

NumPy decoder outputs accumulate on CPU. Ordinary tensor decoder outputs stay on
their device until finalization; accumulation is detached and casts to float64
before arithmetic. Returned distributions keep the existing NumPy format.
Retaining samples explicitly transfers each draw to CPU. This avoids an implicit
per-draw device transfer when only summary statistics are needed.

The benchmark engine remains case-parallel: different ranks evaluate different
cases, and their prediction statistics must not be combined. The adapter therefore
rejects ShardTensor inputs. Application-owned member/domain workflows use the
shared tensor statistics directly, then construct the existing
``FieldDistribution``. Configuring ``statistics_backend`` alone does not change
the benchmark engine's scheduler.

Member/domain parallel example
------------------------------

``workflows/benchmarking/uq_member_parallel.py`` demonstrates a separate same-case
workflow with five members and ten spatial points. With four ranks it uses two
member groups and two uneven spatial partitions (3 and 7 points). One group runs
out of members first and contributes an empty shard while still participating in
collectives. The field statistics retain spatial sharding; only the final
summaries are gathered with ``full_tensor()`` for existing CFD metrics.

.. code-block:: bash

   torchrun --standalone --nproc-per-node=4 workflows/benchmarking/uq_member_parallel.py

Two-rank runs isolate either spatial or member parallelism:

.. code-block:: bash

   torchrun --standalone --nproc-per-node=2 workflows/benchmarking/uq_member_parallel.py --member-groups=1
   torchrun --standalone --nproc-per-node=2 workflows/benchmarking/uq_member_parallel.py --member-groups=2

The example runs with CPU/Gloo or GPU/NCCL and checks analytic reference values,
GP within-/between-member variance, a pressure-derived force observable, existing
coverage metrics, and isolation between different cases. Predictions are already
decoded into physical units. Replace the analytic predictor with model outputs
using the same point ordering on each member group.

The force statistic measures variation of member mean forces. Propagating a GP's
uncertainty to an integrated force additionally needs joint predictive samples or
cross-point covariance; pointwise variances alone are insufficient. This example
does not change the benchmark engine's case-parallel scheduler.
