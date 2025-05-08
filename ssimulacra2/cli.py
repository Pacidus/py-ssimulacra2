#!/usr/bin/env python3
"""Command-line interface for SSIMULACRA2."""

import argparse
import sys
from pathlib import Path

from . import __version__
from .ssimulacra2 import compute_ssimulacra2_with_alpha


def main():
    """Run SSIMULACRA2 from command line."""
    parser = argparse.ArgumentParser(
        description="SSIMULACRA2: Structural SIMilarity Unveiling Local And Compression Related Artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_image", help="Path to original image")
    parser.add_argument("distorted_image", help="Path to distorted image")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed score interpretation",
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"ssymulacra2 {__version__}"
    )

    args = parser.parse_args()

    # Check if files exist
    input_path = Path(args.input_image)
    distorted_path = Path(args.distorted_image)

    if not input_path.exists():
        print(f"Error: Original image '{args.input_image}' not found", file=sys.stderr)
        sys.exit(1)
    if not distorted_path.exists():
        print(
            f"Error: Distorted image '{args.distorted_image}' not found",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        score = compute_ssimulacra2_with_alpha(args.input_image, args.distorted_image)
        print(f"{score:.8f}")

        if args.verbose:
            print("Score interpretation:")
            print("     negative scores: extremely low quality, very strong distortion")
            print(
                "     10 = very low quality (average output of cjxl -d 14 / -q 12 or libjpeg-turbo quality 14, 4:2:0)"
            )
            print(
                "     30 = low quality (average output of cjxl -d 9 / -q 20 or libjpeg-turbo quality 20, 4:2:0)"
            )
            print(
                "     50 = medium quality (average output of cjxl -d 5 / -q 45 or libjpeg-turbo quality 35, 4:2:0)"
            )
            print(
                "     70 = high quality (hard to notice artifacts without comparison to the original)"
            )
            print(
                "     80 = very high quality (impossible to distinguish from the original in a side-by-side comparison at 1:1)"
            )
            print(
                "     85 = excellent quality (impossible to distinguish from the original in a flip test at 1:1)"
            )
            print(
                "     90 = visually lossless (impossible to distinguish from the original in a flicker test at 1:1)"
            )
            print("     100 = mathematically lossless")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
