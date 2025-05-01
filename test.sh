#!/usr/bin/env bash
# SSIMULACRA2 Performance Test Script
# Tests the performance of SSIMULACRA2 implementation using Hyperfine

set -e

# Help message
usage() {
    echo "Usage: $0 <original_image> <compressed_image> [options]"
    echo "Options:"
    echo "  --full                Test all tagged versions"
    echo "  --tag <tag_version>   Test a specific tagged version (e.g., v0.1.0)"
    echo "  --warmup <count>      Number of warmup runs (default: 2)"
    echo "  --runs <count>        Number of benchmark runs (default: 10)"
    echo "  --help                Show this help message"
    exit 1
}

# Check if the required tools are installed
check_requirements() {
    if ! command -v hyperfine &> /dev/null; then
        echo "Error: hyperfine is not installed. Please install it first."
        echo "Installation instructions: https://github.com/sharkdp/hyperfine#installation"
        exit 1
    fi
    
    if ! command -v git &> /dev/null; then
        echo "Error: git is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 is not installed. Please install it first."
        exit 1
    fi
}

# Run performance test for a specific version
run_test() {
    local original_img=$1
    local compressed_img=$2
    local version=$3
    local warmup_runs=$4
    local benchmark_runs=$5
    
    echo "Testing SSIMULACRA2 version: $version"
    
    # Check out the specific version
    git checkout $version 2>/dev/null
    
    # Install the package in development mode
    pip install -e . >/dev/null 2>&1
    
    # Get image dimensions for the report
    img_info=$(python3 -c "from PIL import Image; img = Image.open('$original_img'); print(f'{img.width}x{img.height}')")
    
    echo "Image size: $img_info"
    
    # Run the benchmark
    hyperfine --warmup $warmup_runs --runs $benchmark_runs \
        "python3 -m ssimulacra2.cli '$original_img' '$compressed_img' --quiet" \
        --export-markdown "performance_${version//\//_}.md"
    
    # Extract benchmark results for README update
    result=$(cat "performance_${version//\//_}.md" | grep -A 5 "Time (mean" | tr '\n' ' ' | sed 's/|//g')
    echo
    echo "### $version"
    echo "Size : **$img_info**"
    echo
    echo '```shell'
    echo "$result"
    echo '```'
    echo
}

# Get all available tags
get_tags() {
    git tag -l | sort -V
}

# Main function
main() {
    check_requirements
    
    # Check if we have enough arguments
    if [ $# -lt 2 ]; then
        usage
    fi
    
    ORIGINAL_IMG=$1
    COMPRESSED_IMG=$2
    shift 2
    
    # Check if images exist
    if [ ! -f "$ORIGINAL_IMG" ]; then
        echo "Error: Original image file does not exist: $ORIGINAL_IMG"
        exit 1
    fi
    
    if [ ! -f "$COMPRESSED_IMG" ]; then
        echo "Error: Compressed image file does not exist: $COMPRESSED_IMG"
        exit 1
    fi
    
    # Default values
    FULL=false
    TAG=""
    WARMUP_RUNS=2
    BENCHMARK_RUNS=10
    
    # Parse options
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --full)
                FULL=true
                shift
                ;;
            --tag)
                if [ -z "$2" ]; then
                    echo "Error: --tag option requires a version argument"
                    exit 1
                fi
                TAG="$2"
                shift 2
                ;;
            --warmup)
                if [ -z "$2" ]; then
                    echo "Error: --warmup option requires a count argument"
                    exit 1
                fi
                WARMUP_RUNS="$2"
                shift 2
                ;;
            --runs)
                if [ -z "$2" ]; then
                    echo "Error: --runs option requires a count argument"
                    exit 1
                fi
                BENCHMARK_RUNS="$2"
                shift 2
                ;;
            --help)
                usage
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Save current branch to return to it later
    CURRENT_BRANCH=$(git branch --show-current)
    
    # Create a temporary directory for test results
    TEMP_DIR=$(mktemp -d)
    
    # Run the tests
    if [ "$FULL" = true ]; then
        # Test all tagged versions
        TAGS=$(get_tags)
        if [ -z "$TAGS" ]; then
            echo "No tags found in this repository."
            exit 1
        fi
        
        echo "Testing all tagged versions..."
        echo
        
        for tag in $TAGS; do
            run_test "$ORIGINAL_IMG" "$COMPRESSED_IMG" "$tag" "$WARMUP_RUNS" "$BENCHMARK_RUNS"
        done
    elif [ -n "$TAG" ]; then
        # Test a specific tagged version
        if ! git rev-parse --verify "$TAG" >/dev/null 2>&1; then
            echo "Error: Tag '$TAG' does not exist."
            exit 1
        fi
        
        run_test "$ORIGINAL_IMG" "$COMPRESSED_IMG" "$TAG" "$WARMUP_RUNS" "$BENCHMARK_RUNS"
    else
        # Test current version (without checkout)
        echo "Testing current version..."
        echo
        
        # Get image dimensions for the report
        img_info=$(python3 -c "from PIL import Image; img = Image.open('$ORIGINAL_IMG'); print(f'{img.width}x{img.height}')")
        
        echo "Image size: $img_info"
        
        # Run the benchmark
        hyperfine --warmup $WARMUP_RUNS --runs $BENCHMARK_RUNS \
            "python3 -m ssimulacra2.cli '$ORIGINAL_IMG' '$COMPRESSED_IMG' --quiet" \
            --export-markdown "performance_current.md"
        
        # Extract benchmark results for README update
        result=$(cat "performance_current.md" | grep -A 5 "Time (mean" | tr '\n' ' ' | sed 's/|//g')
        echo
        echo "### current"
        echo "Size : **$img_info**"
        echo
        echo '```shell'
        echo "$result"
        echo '```'
        echo
    fi
    
    # Return to the original branch
    git checkout "$CURRENT_BRANCH" 2>/dev/null
    
    echo "Performance testing completed."
    echo "Results files: performance_*.md"
}

main "$@"
