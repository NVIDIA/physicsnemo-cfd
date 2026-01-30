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

# Base image using NVIDIA CUDA with Ubuntu 22.04
FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Platform architecture detection (amd64 or arm64)
ARG TARGETPLATFORM

# Set OpenFOAM installation directory
ENV OPENFOAM_ROOT=/openfoam
WORKDIR ${OPENFOAM_ROOT}

# Install essential build dependencies and development tools
RUN apt-get update && echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get install -y curl wget build-essential ca-certificates flex libfl-dev bison zlib1g-dev  \
    libboost-system-dev libboost-thread-dev gnuplot libreadline-dev libncurses-dev  \
    libxt-dev libqt5x11extras5-dev qtbase5-dev qt5-qmake qttools5-dev libgmp3-dev git libopenmpi-dev vim

# Install CMake based on target architecture (x86_64 or aarch64)
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        # Setup for x86_64 architecture
        export CMAKE=${OPENFOAM_ROOT}/cmake-3.23.2-linux-x86_64/bin/cmake && \
        export PATH=${PATH}:${OPENFOAM_ROOT}/cmake-3.23.2-linux-x86_64/bin && \
        mkdir cmake-3.23.2-linux-x86_64 && cd cmake-3.23.2-linux-x86_64 && \
        wget https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-linux-x86_64.sh && \
        chmod +x cmake-3.23.2-linux-x86_64.sh && ./cmake-3.23.2-linux-x86_64.sh --skip-license &&  ls cmake* && \
        export PATH=$PATH:${OPENFOAM_ROOT}/cmake-3.23.2-linux-x86_64/bin && cmake --version ; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        # Setup for ARM64 architecture
        export CMAKE=${OPENFOAM_ROOT}/cmake-3.23.2-linux-aarch64/bin/cmake && \
        export PATH=${PATH}:${OPENFOAM_ROOT}/cmake-3.23.2-linux-aarch64/bin && \
        mkdir cmake-3.23.2-linux-aarch64 && cd cmake-3.23.2-linux-aarch64 && \
        wget https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2-linux-aarch64.sh && \
        chmod +x cmake-3.23.2-linux-aarch64.sh && ./cmake-3.23.2-linux-aarch64.sh --skip-license &&  ls cmake* && \
        export PATH=$PATH:${OPENFOAM_ROOT}/cmake-3.23.2-linux-aarch64/bin && cmake --version ; \
    fi

# -------------------------------------------------------
# OpenFOAM stage - download and build OpenFOAM
# -------------------------------------------------------
FROM builder AS openfoam-install

ARG TARGETPLATFORM

ENV OPENFOAM_ROOT=/openfoam
WORKDIR ${OPENFOAM_ROOT}

# Download OpenFOAM and ThirdParty source packages
RUN cd ${OPENFOAM_ROOT} && wget https://dl.openfoam.com/source/v2206/OpenFOAM-v2206.tgz && \
    wget https://dl.openfoam.com/source/v2206/ThirdParty-v2206.tgz

# Extract OpenFOAM and ThirdParty archives
RUN tar -zxvf OpenFOAM-v2206.tgz && tar -zxvf ThirdParty-v2206.tgz

# Use bash for subsequent RUN commands (needed for OpenFOAM's bashrc)
SHELL ["/bin/bash", "-c"]

# Build OpenFOAM with appropriate architecture settings
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
        # Build for x86_64
        export CMAKE=${OPENFOAM_ROOT}/cmake-3.23.2-linux-x86_64/bin/cmake && \
        export PATH=${PATH}:${OPENFOAM_ROOT}/cmake-3.23.2-linux-x86_64/bin && \
        source $OPENFOAM_ROOT/OpenFOAM-v2206/etc/bashrc && cd $WM_PROJECT_DIR && ./Allwmake -j -s -l -q -k; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        # Build for ARM64
        export CMAKE=${OPENFOAM_ROOT}/cmake-3.23.2-linux-aarch64/bin/cmake && \
        export PATH=${PATH}:${OPENFOAM_ROOT}/cmake-3.23.2-linux-aarch64/bin && \
        source $OPENFOAM_ROOT/OpenFOAM-v2206/etc/bashrc && cd $WM_PROJECT_DIR && ./Allwmake -j -s -l -q -k; \
    fi

# # -------------------------------------------------------
# # PETSc stage - Install PETSc (Portable, Extensible Toolkit for Scientific Computation)
# # -------------------------------------------------------
# FROM openfoam-install AS petsc-install

# ARG TARGETPLATFORM

# ENV OPENFOAM_ROOT=/openfoam
# WORKDIR ${OPENFOAM_ROOT}

# # Clone PETSc repository
# RUN git clone -b release https://gitlab.com/petsc/petsc.git petsc

# # Update the PETSc version in the OpenFOAM configuration to match the one we're building
# RUN sed -i 's/petsc_version=petsc-3.17.2/petsc_version=petsc/' $OPENFOAM_ROOT/OpenFOAM-v2206/etc/config.sh/petsc

# # Configure and build PETSc for the appropriate architecture
# RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
#         # Build PETSc for x86_64
#         export CMAKE=${OPENFOAM_ROOT}/cmake-3.23.2-linux-x86_64/bin/cmake && \
#         export PATH=${PATH}:${OPENFOAM_ROOT}/cmake-3.23.2-linux-x86_64/bin && \
#         cd $OPENFOAM_ROOT/petsc/ && source $OPENFOAM_ROOT/OpenFOAM-v2206/etc/bashrc && \
#             ./configure --with-64-bit-indices=0 \
#             --with-precision=double \
#             --prefix=$WM_THIRD_PARTY_DIR/platforms/$WM_ARCH$WM_COMPILER$WM_PRECISION_OPTION$WM_LABEL_OPTION/petsc  \
#             PETSC_ARCH=$WM_OPTIONS \
#             --download-f2cblaslapack \
#             --with-fc=0 \
#             --force && make all && make install && cd $OPENFOAM_ROOT/; \
#     elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
#         # Build PETSc for ARM64
#         export CMAKE=${OPENFOAM_ROOT}/cmake-3.23.2-linux-aarch64/bin/cmake && \
#         export PATH=${PATH}:${OPENFOAM_ROOT}/cmake-3.23.2-linux-aarch64/bin && \
#         cd $OPENFOAM_ROOT/petsc/ && source $OPENFOAM_ROOT/OpenFOAM-v2206/etc/bashrc && \
#             ./configure --with-64-bit-indices=0 \
#             --with-precision=double \
#             --prefix=$WM_THIRD_PARTY_DIR/platforms/$WM_ARCH$WM_COMPILER$WM_PRECISION_OPTION$WM_LABEL_OPTION/petsc  \
#             PETSC_ARCH=$WM_OPTIONS \
#             --download-f2cblaslapack \
#             --with-fc=0 \
#             --force && make all && make install && cd $OPENFOAM_ROOT/; \
#     fi

# # -------------------------------------------------------
# # foam2csr stage - Build foam2csr converter for OpenFOAM matrices
# # -------------------------------------------------------
# FROM petsc-install AS foam2csr-install

# ARG TARGETPLATFORM

# ENV OPENFOAM_ROOT=/openfoam
# WORKDIR ${OPENFOAM_ROOT}

# # Set environment variables for foam2csr build
# ENV AMGX_INC=$OPENFOAM_ROOT/amgx/install/include/
# ENV AMGX_LIB=$OPENFOAM_ROOT/amgx/install/lib/
# ENV SPECTRUM_MPI_HOME=.
# ENV CUBROOT=$OPENFOAM_ROOT/petsc/include/

# # Clone foam2csr repository
# RUN git clone https://gitlab.hpc.cineca.it/openfoam/foam2csr.git

# # Build foam2csr for the appropriate architecture
# RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
#         # Build foam2csr for x86_64
#         export CPATH=/usr/lib/x86_64-linux-gnu/openmpi/include/:$CPATH && \
#         export CMAKE=${OPENFOAM_ROOT}/cmake-3.23.2-linux-x86_64/bin/cmake && \
#         export PATH=${PATH}:${OPENFOAM_ROOT}/cmake-3.23.2-linux-x86_64/bin && \
#         export PETSC_LIB_DIR=$OPENFOAM_ROOT/petsc/linux64GccDPInt32Opt/lib/ && \
#         export PETSC_INC=$OPENFOAM_ROOT/petsc/linux64GccDPInt32Opt/include/ && \
#         source $OPENFOAM_ROOT/OpenFOAM-v2206/etc/bashrc && cd $OPENFOAM_ROOT/foam2csr && ./Allwclean && ./Allwmake -j && cd $OPENFOAM_ROOT ;\
#     elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
#         # Build foam2csr for ARM64
#         export CPATH=/usr/lib/aarch64-linux-gnu/openmpi/include/:$CPATH && \
#         export CMAKE=${OPENFOAM_ROOT}/cmake-3.23.2-linux-aarch64/bin/cmake && \
#         export PATH=${PATH}:${OPENFOAM_ROOT}/cmake-3.23.2-linux-aarch64/bin && \
#         export PETSC_LIB_DIR=$OPENFOAM_ROOT/petsc/linuxARM64GccDPInt32Opt/lib/ && \
#         export PETSC_INC=$OPENFOAM_ROOT/petsc/linuxARM64GccDPInt32Opt/include/ && \
#         source $OPENFOAM_ROOT/OpenFOAM-v2206/etc/bashrc && cd $OPENFOAM_ROOT/foam2csr && ./Allwclean && ./Allwmake -j && cd $OPENFOAM_ROOT ;\
#     fi

# # -------------------------------------------------------
# # petsc4foam stage - Build the PETSc interface for OpenFOAM
# # -------------------------------------------------------
# FROM foam2csr-install AS petsc4foam-install

# ARG TARGETPLATFORM

# ENV OPENFOAM_ROOT=/openfoam
# WORKDIR ${OPENFOAM_ROOT}

# # Set environment variables for petsc4foam build
# ENV FOAM2CSR_INC=$OPENFOAM_ROOT/foam2csr/src/
# ENV CPATH=/usr/local/cuda/include/:$CPATH
# ENV INCLUDE_PATH=/usr/local/cuda/include/:$INCLUDE_PATH

# # Clone petsc4foam repository with AMGX wrapper support
# RUN git clone --branch amgxwrapper https://develop.openfoam.com/modules/external-solver.git petsc4foam

# # Build petsc4foam for the appropriate architecture
# RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
#         # Build petsc4foam for x86_64
#         export CPATH=/usr/lib/x86_64-linux-gnu/openmpi/include/:$CPATH && \
#         export CMAKE=${OPENFOAM_ROOT}/cmake-3.23.2-linux-x86_64/bin/cmake && \
#         export PATH=${PATH}:${OPENFOAM_ROOT}/cmake-3.23.2-linux-x86_64/bin && \
#         source $OPENFOAM_ROOT/OpenFOAM-v2206/etc/bashrc && cd petsc4foam && ./Allwclean && ./Allwmake -j && cd $OPENFOAM_ROOT ;\
#     elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
#         # Build petsc4foam for ARM64
#         export CPATH=/usr/lib/aarch64-linux-gnu/openmpi/include/:$CPATH && \
#         export CMAKE=${OPENFOAM_ROOT}/cmake-3.23.2-linux-aarch64/bin/cmake && \
#         export PATH=${PATH}:${OPENFOAM_ROOT}/cmake-3.23.2-linux-aarch64/bin && \
#         source $OPENFOAM_ROOT/OpenFOAM-v2206/etc/bashrc && cd petsc4foam && ./Allwclean && ./Allwmake -j && cd $OPENFOAM_ROOT ;\
#     fi

# -------------------------------------------------------
# Python install and dependencies
# -------------------------------------------------------
FROM openfoam-install AS python-install

# Combine all Python-related operations in single RUN
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libfreetype6-dev libpng-dev libgl1-mesa-dev xvfb && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir numpy pyvista tqdm numba scikit-learn scipy matplotlib warp-lang trimesh && \
    find /usr/local -type d -name '__pycache__' -exec rm -rf {} + && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# -------------------------------------------------------
# Final setup
# -------------------------------------------------------

# Clean up build files
RUN rm -rf \
    ${OPENFOAM_ROOT}/OpenFOAM-v2206.tgz \
    ${OPENFOAM_ROOT}/ThirdParty-v2206.tgz
    # ${OPENFOAM_ROOT}/petsc/build \
    # ${OPENFOAM_ROOT}/foam2csr/platforms/*/src

# Copy configuration script to container
COPY configs.sh /openfoam/

# Make configuration script executable
RUN chmod 777 /openfoam/configs.sh

# Set the container's entry point
CMD ["/openfoam/configs.sh"] 
