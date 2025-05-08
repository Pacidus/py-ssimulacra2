# SSIMULACRA2

[![PyPI - Version](https://img.shields.io/pypi/v/ssimulacra2.svg)](https://pypi.org/project/ssimulacra2)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ssimulacra2.svg)](https://pypi.org/project/ssimulacra2)

A Python implementation of SSIMULACRA2 (Structural SIMilarity Unveiling Local And Compression Related Artifacts) - a perceptual image quality metric that helps you detect and measure compression artifacts.

## What is SSIMULACRA2?

SSIMULACRA2 is a full-reference image quality metric designed to mimic how humans perceive image quality, with a special focus on compression artifacts. This Python package offers a clean and efficient implementation that closely follows the original C++ algorithm from the JPEG XL project.

Want to know if your compressed images still look good? SSIMULACRA2 gives you a score from 100 (perfect quality) down to negative values (terrible quality) to help you decide:

- **Negative scores**: You've gone too far! Extremely low quality with very strong distortion
- **10**: Very low quality - comparable to what you'd get from cjxl -d 14 / -q 12 or libjpeg-turbo quality 14, 4:2:0
- **30**: Low quality - similar to output from cjxl -d 9 / -q 20 or libjpeg-turbo quality 20, 4:2:0
- **50**: Medium quality - like what cjxl -d 5 / -q 45 or libjpeg-turbo quality 35, 4:2:0 would produce
- **70**: High quality - you'd have trouble noticing artifacts without comparing to the original
- **80**: Very high quality - most people couldn't tell the difference from the original in a side-by-side comparison
- **85**: Excellent quality - virtually impossible to distinguish from the original in a flip test
- **90**: Visually lossless - even in a flicker test at 1:1, you can't tell the difference
- **100**: Mathematically lossless - pixel-perfect match to the original

## Getting Started

### Installation

It's easy to install via pip:

```console
pip install ssimulacra2
```

### Usage

#### Command Line

Simple and straightforward:

```console
# Basic usage - get the score with interpretation
ssymulacra2 original.png compressed.png

# Just the score, no extra info
ssymulacra2 original.png compressed.png --quiet
```

#### Python

Import directly into your Python projects:

```python
from ssimulacra2 import compute_ssimulacra2, compute_ssimulacra2_with_alpha

# Basic usage
score = compute_ssimulacra2("original.png", "compressed.png")
print(f"Quality score: {score:.2f}")

# For images with alpha channel (automatically uses both dark and light backgrounds)
score = compute_ssimulacra2_with_alpha("original.png", "compressed.png")
print(f"Quality score with alpha: {score:.2f}")
```

## Goal

This implementation aims to closely match the original C++ algorithm, providing results consistent with the reference code.

## Requirements

You'll need:
- Python 3.8+
- NumPy
- SciPy
- Pillow (PIL)

## Performance

Performance benchmarks for a 1024x768 image:

| Version | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `v0.1.0` | 22.456 ± 0.144 | 22.245 | 22.680 | 33.29 ± 0.59 |
| `v0.2.0` | 0.674 ± 0.011 | 0.661 | 0.696 | 1.00 |
| `HEAD` | 0.689 ± 0.028 | 0.666 | 0.764 | 1.02 ± 0.05 |

The dramatic speed improvement from v0.1.0 to v0.2.0 comes from better leveraging NumPy's vectorized operations. Note that the slight differences in results between versions are due to ongoing experimentation with Gaussian blur parameters to better match the original implementation.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Acknowledgements

This implementation is based on the original SSIMULACRA2 algorithm developed for the JPEG XL project.
