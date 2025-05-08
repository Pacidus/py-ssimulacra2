# SSIMULACRA2

[![PyPI - Version](https://img.shields.io/pypi/v/ssimulacra2.svg)](https://pypi.org/project/ssimulacra2)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ssimulacra2.svg)](https://pypi.org/project/ssimulacra2)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A Python implementation of SSIMULACRA2 (Structural SIMilarity Unveiling Local And Compression Related Artifacts) - a perceptual image quality metric designed to detect and measure compression artifacts.

## Overview

SSIMULACRA2 is a full-reference image quality metric that mimics human perception of image quality, focusing specifically on compression artifacts. This Python package provides an efficient implementation that closely follows the original C++ algorithm from the JPEG XL project.

### Quality Score Interpretation

SSIMULACRA2 scores range from 100 (perfect quality) down to negative values (severe degradation):

| Score | Quality Level | Description | Example Comparison |
|-------|---------------|-------------|-------------------|
| < 0 | Extremely Low | Very strong distortion | - |
| 10 | Very Low | Heavy artifacts | cjxl -d 14 / -q 12 or libjpeg-turbo quality 14, 4:2:0 |
| 30 | Low | Noticeable artifacts | cjxl -d 9 / -q 20 or libjpeg-turbo quality 20, 4:2:0 |
| 50 | Medium | Acceptable quality | cjxl -d 5 / -q 45 or libjpeg-turbo quality 35, 4:2:0 |
| 70 | High | Hard to notice artifacts without comparison | - |
| 80 | Very High | Difficult to distinguish in side-by-side comparison | - |
| 85 | Excellent | Virtually indistinguishable in flip tests | - |
| 90 | Visually Lossless | Imperceptible differences even in flicker tests | - |
| 100 | Mathematically Lossless | Pixel-perfect match | - |

## Installation

```console
pip install ssimulacra2
```

## Usage

### Command Line

```console
# Basic usage with detailed quality interpretation
ssymulacra2 original.png compressed.png

# Just the score, no extra info
ssymulacra2 original.png compressed.png --quiet
```

### Python API

```python
from ssimulacra2 import compute_ssimulacra2, compute_ssimulacra2_with_alpha

# Basic usage
score = compute_ssimulacra2("original.png", "compressed.png")
print(f"Quality score: {score:.2f}")

# For images with alpha channel (automatically uses both dark and light backgrounds)
score = compute_ssimulacra2_with_alpha("original.png", "compressed.png")
print(f"Quality score with alpha: {score:.2f}")
```

## Performance

This implementation is optimized for speed while maintaining accuracy. Performance benchmarks for a 1024x768 image:

| Version | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `v0.1.0` | 22.456 ± 0.144 | 22.245 | 22.680 | 33.29 ± 0.59 |
| `v0.2.0` | 0.674 ± 0.011 | 0.661 | 0.696 | 1.00 |
| `HEAD` | 0.689 ± 0.028 | 0.666 | 0.764 | 1.02 ± 0.05 |

The dramatic speed improvement from v0.1.0 to v0.2.0 comes from better leveraging NumPy's vectorized operations.

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Pillow (PIL)

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Acknowledgements

This implementation is based on the original SSIMULACRA2 algorithm developed for the JPEG XL project.
