# SSIMULACRA2

[![PyPI - Version](https://img.shields.io/pypi/v/ssimulacra2.svg)](https://pypi.org/project/ssimulacra2)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ssimulacra2.svg)](https://pypi.org/project/ssimulacra2)

Python implementation of SSIMULACRA2 (Structural SIMilarity Unveiling Local And Compression Related Artifacts) - a perceptual image quality metric designed to accurately detect compression artifacts.

## Description

SSIMULACRA2 is a full-reference image quality metric that aims to estimate human perception of image quality, specifically focusing on compression artifacts. This Python package provides a clean, efficient implementation that closely follows the original C++ algorithm.

SSIMULACRA2 scores range from 100 (perfect quality) down to negative values (extremely poor quality):

- **negative scores**: extremely low quality, very strong distortion
- **10**: very low quality (average output of cjxl -d 14 / -q 12 or libjpeg-turbo quality 14, 4:2:0)
- **30**: low quality (average output of cjxl -d 9 / -q 20 or libjpeg-turbo quality 20, 4:2:0)
- **50**: medium quality (average output of cjxl -d 5 / -q 45 or libjpeg-turbo quality 35, 4:2:0)
- **70**: high quality (hard to notice artifacts without comparison to the original)
- **80**: very high quality (impossible to distinguish from the original in a side-by-side comparison at 1:1)
- **85**: excellent quality (impossible to distinguish from the original in a flip test at 1:1)
- **90**: visually lossless (impossible to distinguish from the original in a flicker test at 1:1)
- **100**: mathematically lossless

## Installation

```console
pip install ssimulacra2
```

## Usage

### Command Line

```console
# Basic usage
ssimulacra2 original.png compressed.png

# For just the score without interpretation
ssimulacra2 original.png compressed.png --quiet
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

## Features

- **Accurate**: Closely follows the original C++ implementation
- **Alpha Support**: Special handling for images with transparency
- **Multi-scale Analysis**: Examines image quality at multiple resolution scales
- **XYB Color Space**: Uses a perceptual color space for more accurate results
- **Easy to Use**: Simple command-line interface and Python API

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Pillow (PIL)
- scikit-image
- OpenCV

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Acknowledgements

This implementation is based on the original SSIMULACRA2 algorithm developed for the JPEG XL project.
