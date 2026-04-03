# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM (NVIDIA Inference Microservice) helpers for evaluation workflows (e.g. DoMINO automotive aero)."""

from physicsnemo.cfd.evaluation.nims.domino_nim import call_domino_nim

__all__ = ["call_domino_nim"]
