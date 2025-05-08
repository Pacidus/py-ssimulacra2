# SPDX-FileCopyrightText: 2025-present Pacidus <pacidus@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""SSIMULACRA2: Structural SIMilarity Unveiling Local And Compression Related Artifacts."""

__version__ = "0.2.2"

from .ssimulacra2 import (
    compute_ssimulacra2,
    compute_ssimulacra2_with_alpha,
)

__all__ = [
    "compute_ssimulacra2",
    "compute_ssimulacra2_with_alpha",
]
