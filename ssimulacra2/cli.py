"""
Command-line interface for the SSIMULACRA2 tool.
"""

import argparse
import os
import sys
import time
from PIL import Image
import json
from .ssimulacra2 import calculate_ssimulacra2


def main():
    """
    Main function for the CLI.
    """
    parser = argparse.ArgumentParser(
        description="Calculate SSIMULACRA2 score between reference and distorted images"
    )

    parser.add_argument("reference", help="Path to the reference image")

    parser.add_argument("distorted", nargs="+", help="Path to the distorted image(s)")

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument("-o", "--output", help="Output file for results (JSON format)")

    args = parser.parse_args()

    # Check if reference file exists
    if not os.path.isfile(args.reference):
        print(f"Error: Reference file '{args.reference}' not found")
        sys.exit(1)

    # Load reference image
    try:
        ref_img = Image.open(args.reference)
    except Exception as e:
        print(f"Error loading reference image: {e}")
        sys.exit(1)

    # Process each distorted image
    results = {}

    for dist_path in args.distorted:
        if not os.path.isfile(dist_path):
            print(f"Warning: Distorted file '{dist_path}' not found, skipping")
            continue

        try:
            if args.verbose:
                print(f"Processing: {dist_path}")
                start_time = time.time()

            # Load distorted image
            dist_img = Image.open(dist_path)

            # Calculate score
            score = calculate_ssimulacra2(ref_img, dist_img)

            # Store result
            results[dist_path] = score

            if args.verbose:
                elapsed = time.time() - start_time
                print(f"  SSIMULACRA2 score: {score:.6f} (computed in {elapsed:.2f}s)")
            else:
                print(f"{dist_path}: {score:.6f}")

        except Exception as e:
            print(f"Error processing '{dist_path}': {e}")

    # Write results to output file if specified
    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(
                    {"reference": args.reference, "results": results}, f, indent=2
                )

            if args.verbose:
                print(f"Results written to: {args.output}")
        except Exception as e:
            print(f"Error writing output file: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
