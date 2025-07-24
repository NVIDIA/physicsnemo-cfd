#!/bin/bash

# Hybrid Initialization Workflow Script
# -------------------------------------
# This script orchestrates the full hybrid initialization workflow for
# automotive external aerodynamics using OpenFOAM and ML-based DoMINO predictions.
# It is intended to be run inside the provided OpenFOAM+Python container.
#
# The script will:
#   1. Clean the working directory
#   2. Prepare DoMINO assets (geometry, initial conditions, ML predictions)
#   3. Generate the mesh
#   4. Compute the potential flow solution
#   5. Create the hybrid initialization
#   6. Run the transient CFD simulation
#   7. Generate VTK output for visualization
#
# Logs for each major step are written to log.* files in the working directory.
#
# Usage:
#   ./run.sh
#
# Notes:
#   - The script will use all available physical CPU cores for parallel steps.
#   - If interrupted during the transient simulation, post-processing will still run.
#   - For troubleshooting, inspect the log.* files.

set -exo pipefail

# Helper: Print error and exit if a required command is missing
require_command() {
    command -v "$1" >/dev/null 2>&1 || { echo "Error: '$1' is required but not installed." >&2; exit 1; }
}

# Check for required commands
for cmd in lscpu mpirun python3 pip3 tee sed grep sort wc blockMesh surfaceFeatureExtract snappyHexMesh checkMesh reconstructParMesh reconstructPar decomposePar potentialFoam pimpleFoam foamToVTK; do
    require_command "$cmd"
done

# Clean previous results
echo "Cleaning previous results..."
./clean.sh

# Prepare DoMINO assets
echo "Preparing DoMINO assets..."
pip3 install --quiet --disable-pip-version-check --root-user-action=ignore httpx
python3 prepare_domino_assets.py | tee log.prepare_domino_assets

# Copy initial conditions and geometry
echo "Copying initial conditions and geometry..."
mkdir -p 0_org/include constant/triSurface
cp from_domino/initialConditions 0_org/include/
cp from_domino/vehicle.stl constant/triSurface/

# Get the number of physical CPU cores (not including hyperthreading)
# For Ubuntu/Linux systems
procs=$(lscpu -p | grep -v '^#' | sort -u -t, -k 2,2 | wc -l)
echo "Detected physical CPU cores: $procs"

# Update numberOfSubdomains in decomposeParDict
if grep -q "numberOfSubdomains" system/decomposeParDict; then
    sed -i "s/numberOfSubdomains [0-9]\+;/numberOfSubdomains $procs;/" system/decomposeParDict
    echo "Updated system/decomposeParDict to use $procs subdomains."
else
    echo "Warning: numberOfSubdomains not found in system/decomposeParDict."
fi

# Prepare 0 directory for potentialFoam
rm -rf 0
cp -r 0_org 0

# Mesh generation
echo "Running mesh generation..."
surfaceFeatureExtract | tee log.surfaceFeatures
blockMesh | tee log.blockMesh
decomposePar
mpirun -np $procs --allow-run-as-root snappyHexMesh -parallel -overwrite | tee log.shm
mpirun -np $procs --allow-run-as-root checkMesh -parallel | tee log.checkmesh
reconstructParMesh -constant
rm -r processor*

# PotentialFoam run
echo "Running potentialFoam..."
decomposePar -force
mpirun -np $procs --allow-run-as-root potentialFoam -parallel -writephi -writePhi -writep | tee log.potentialfoam
reconstructPar -withZero
rm -r processor*

# Stitch together the potential flow and ML meshes using Python
echo "Stitching potential flow and ML meshes..."
cp -r 0/ 0_pf/
cp -r 0_org 0_hybrid
python3 make_hybrid_initialization.py | tee log.make_hybrid_initialization
cp -r 0_hybrid/ 0/

# PimpleFoam run
echo "Running PimpleFoam..."
decomposePar -force
mpirun -np $procs --allow-run-as-root pimpleFoam -parallel | tee log.pimplefoam
reconstructPar -latestTime
rm -r processor*

# VTK output
echo "Generating VTK output..."
foamToVTK -latestTime

echo "Workflow completed successfully!"