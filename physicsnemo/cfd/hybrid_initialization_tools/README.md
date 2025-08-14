# Hybrid Initialization Tools

This directory contains:

- A Python submodule with tools for creating hybrid initializations for
  transient CFD simulations, following the methods in [this
  paper](https://arxiv.org/abs/2503.15766). These "hybrid initializations"
  combine a potential flow solution with an ML surrogate initialization (e.g.,
  from a [DoMINO
  NIM](https://docs.nvidia.com/nim/physicsnemo/domino-automotive-aero/latest/overview.html)).
  This package does not contain either a) the ML model or b) the potential flow
  solver, in order to be as general as possible.

For example usage, see `/workflows/hybrid_initialization_example/`, which provides
a self-contained example for a transient automotive aerodynamics case using
OpenFOAM-based solvers.
