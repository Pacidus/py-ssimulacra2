# SSIMULACRA2

[![PyPI - Version](https://img.shields.io/pypi/v/ssimulacra2.svg)](https://pypi.org/project/ssimulacra2)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ssimulacra2.svg)](https://pypi.org/project/ssimulacra2)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A Python implementation of SSIMULACRA2 (Structural SIMilarity Unveiling Local And Compression Related Artifacts) - a perceptual image quality metric designed to detect and measure compression artifacts.

## Overview

SSIMULACRA2 is a full-reference image quality metric that mimics human perception of image quality, focusing specifically on compression artifacts. This Python package provides an efficient implementation that closely follows the original C++ algorithm from the JPEG XL project.

For detailed information about the metric and score interpretation, please refer to the [original SSIMULACRA2 implementation](https://github.com/libjxl/libjxl/tree/main/tools/ssimulacra2) in the JPEG XL repository.

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

## Performance and Benchmarking

### Performance Metrics

This implementation is optimized for speed while maintaining accuracy. Performance benchmarks for a 1024x768 image:

| Version | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `v0.1.0` | 22.456 ± 0.144 | 22.245 | 22.680 | 33.29 ± 0.59 |
| `v0.2.0` | 0.674 ± 0.011 | 0.661 | 0.696 | 1.00 |
| `HEAD` | 0.689 ± 0.028 | 0.666 | 0.764 | 1.02 ± 0.05 |

The dramatic speed improvement from v0.1.0 to v0.2.0 comes from better leveraging NumPy's vectorized operations.

### Running Your Own Benchmarks

The package includes a performance benchmarking script that allows you to evaluate the speed of different versions:

```bash
# Benchmark current version
./performance_benchmark.sh original.png compressed.png

# Benchmark a specific version
./performance_benchmark.sh original.png compressed.png --tag v0.2.0

# Compare all available versions
./performance_benchmark.sh original.png compressed.png --full

# Customize benchmark parameters
./performance_benchmark.sh original.png compressed.png --warmup 5 --runs 20
```

#### Benchmark Script Options

```
Usage: ./performance_benchmark.sh <original_image> <compressed_image> [options]
Options:
  --full                Test all tagged versions simultaneously
  --tag <tag_version>   Test a specific tagged version (e.g., v0.1.0)
  --warmup <count>      Number of warmup runs (default: 2)
  --runs <count>        Number of benchmark runs (default: 10)
  --debug               Display debugging information
  --help                Show this help message
```

### Using with press_it Benchmarks

SSIMULACRA2 is used by the [press_it](https://github.com/Pacidus/press_it) package's benchmark tool to evaluate image compression quality across different formats. When running `press-benchmark`, this Python implementation is used to calculate quality scores for compressed images (alongside C++ and Rust implementations when available).

For more comprehensive compression benchmarking, refer to the press_it documentation.

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Pillow (PIL)

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Acknowledgements

This implementation is based on the original SSIMULACRA2 algorithm developed for the JPEG XL project.
