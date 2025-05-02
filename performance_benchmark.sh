#!/usr/bin/env bash
# SSIMULACRA2 Performance Benchmark Script
# Tests the performance of SSIMULACRA2 implementation using Hyperfine

set -e

# Help message
usage() {
    echo "Usage: $0 <original_image> <compressed_image> [options]"
    echo "Options:"
    echo "  --full                Test all tagged versions simultaneously"
    echo "  --tag <tag_version>   Test a specific tagged version (e.g., v0.1.0)"
    echo "  --warmup <count>      Number of warmup runs (default: 2)"
    echo "  --runs <count>        Number of benchmark runs (default: 10)"
    echo "  --debug               Display debugging information"
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

# Get all available tags
get_tags() {
    git tag -l | sort -V
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

# Test a specific version wrapper script
test_wrapper_script() {
    local script_path=$1
    local debug=$2
    local version=$3
    
    # Test if the script works
    if [ "$debug" = true ]; then
        echo "Testing wrapper script for $version: $script_path"
        bash -x "$script_path"
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "Error: Wrapper script for $version failed with exit code $exit_code"
        fi
        return $exit_code
    else
        # Even in non-debug mode, capture and log errors
        echo "Testing wrapper script for $version..."
        output=$(bash "$script_path" 2>&1)
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "Error: Wrapper script for $version failed with exit code $exit_code"
            echo "Error output: $output"
            echo "Try running with --debug for more information"
            return 1
        fi
    fi
    
    return 0
}

# Setup a temporary environment for running a specific version
setup_version_env() {
    local version=$1
    local temp_dir=$2
    local original_img=$3
    local compressed_img=$4
    local current_branch=$5
    local debug=$6
    local shared_dir=$7
    
    echo "Setting up environment for $version..."
    
    # Create a temporary directory for this version
    mkdir -p "$temp_dir/$version"
    
    # Clone the repository to the temporary directory
    git clone . "$temp_dir/$version" 2>/dev/null || {
        echo "Error: Failed to clone repository for $version"
        return 1
    }
    
    # Change to the temporary directory
    pushd "$temp_dir/$version" > /dev/null || {
        echo "Error: Failed to change directory to $temp_dir/$version"
        return 1
    }
    
    # For current version, use the current branch instead of a tag
    if [ "$version" = "HEAD" ]; then
        # Use the current commit
        git checkout "$current_branch" 2>/dev/null || {
            echo "Error: Failed to checkout current branch $current_branch"
            popd > /dev/null
            return 1
        }
    else
        # Check out the specific version
        git checkout "$version" 2>/dev/null || {
            echo "Error: Failed to checkout version $version"
            popd > /dev/null
            return 1
        }
    fi
    
    # Install the package in development mode
    echo "Installing package for $version..."
    if [ "$debug" = true ]; then
        pip install -e . || {
            echo "Error: Failed to install package for $version"
            popd > /dev/null
            return 1
        }
    else
        pip install -e . >/dev/null 2>&1 || {
            echo "Error: Failed to install package for $version"
            popd > /dev/null
            return 1
        }
    fi
    
    # Create a wrapper script that runs this version
    script_name="run_${version//\//_}.sh"
    script_path="$temp_dir/$version/$script_name"
    
    cat > "$script_path" << EOF
#!/bin/bash
cd "$temp_dir/$version" || exit 1
python3 -m ssimulacra2.cli "$shared_dir/original.png" "$shared_dir/compressed.png" --quiet
EOF
    
    chmod +x "$script_path" || {
        echo "Error: Failed to make script executable for $version"
        popd > /dev/null
        return 1
    }
    
    # Test if the wrapper script works
    if ! test_wrapper_script "$script_path" "$debug" "$version"; then
        echo "Warning: Wrapper script for $version failed. Skipping this version."
        popd > /dev/null
        return 1
    fi
    
    echo "Setup for $version completed successfully"
    
    # Return the path to the wrapper script
    echo "$script_path"
    
    popd > /dev/null
    return 0
}

# Run full comparison test across all versions
run_full_comparison() {
    local original_img=$1
    local compressed_img=$2
    local warmup_runs=$3
    local benchmark_runs=$4
    local temp_dir=$5
    local current_branch=$6
    local debug=$7
    
    # Create a shared directory for the test images
    local shared_dir="$temp_dir/shared"
    mkdir -p "$shared_dir"
    
    # Copy the test images to the shared directory with standard names
    echo "Copying test images to shared directory..."
    cp "$original_img" "$shared_dir/original.png"
    cp "$compressed_img" "$shared_dir/compressed.png"
    
    if [ ! -f "$shared_dir/original.png" ] || [ ! -f "$shared_dir/compressed.png" ]; then
        echo "Error: Failed to copy test images to shared directory"
        exit 1
    fi
    
    # Get all tags
    TAGS=$(get_tags)
    if [ -z "$TAGS" ]; then
        echo "No tags found in this repository."
        exit 1
    fi
    
    echo "Setting up environments for all tagged versions..."
    
    # Get image dimensions for the report
    img_info=$(python3 -c "from PIL import Image; img = Image.open('$shared_dir/original.png'); print(f'{img.width}x{img.height}')")
    echo "Image size: $img_info"
    
    # Build the hyperfine command with arrays
    declare -a cmd_names=()
    declare -a cmd_scripts=()
    
    # Setup environments for each version and add to the command arrays
    for tag in $TAGS; do
        echo "Setting up $tag..."
        script_path=$(setup_version_env "$tag" "$temp_dir" "$original_img" "$compressed_img" "$current_branch" "$debug" "$shared_dir")
        if [ $? -eq 0 ] && [ -n "$script_path" ]; then
            cmd_names+=("$tag")
            cmd_scripts+=("$script_path")
            echo "Successfully set up $tag"
        else
            echo "Skipping $tag due to setup failure"
        fi
    done
    
    # Add current version (HEAD)
    echo "Setting up HEAD (current version)..."
    script_path=$(setup_version_env "HEAD" "$temp_dir" "$original_img" "$compressed_img" "$current_branch" "$debug" "$shared_dir")
    if [ $? -eq 0 ] && [ -n "$script_path" ]; then
        cmd_names+=("HEAD")
        cmd_scripts+=("$script_path")
        echo "Successfully set up HEAD"
    else
        echo "Skipping HEAD due to setup failure"
    fi
    
    # Check if we have any commands to run
    if [ ${#cmd_names[@]} -eq 0 ]; then
        echo "Error: No valid versions to benchmark. Please check the setup."
        exit 1
    fi
    
    # Build the final hyperfine command
    hyperfine_cmd="hyperfine --warmup $warmup_runs --runs $benchmark_runs --ignore-failure"
    
    # Add all commands to hyperfine
    for i in "${!cmd_names[@]}"; do
        hyperfine_cmd="$hyperfine_cmd --command-name \"${cmd_names[$i]}\" \"${cmd_scripts[$i]}\""
    done
    
    # Add export option
    hyperfine_cmd="$hyperfine_cmd --export-markdown \"performance_comparison.md\""
    
    # Run the comparison
    echo -e "\nRunning performance comparison across all versions..."
    echo "Command: $hyperfine_cmd"
    
    # Run the command with error handling
    if ! eval "$hyperfine_cmd"; then
        echo "Error running hyperfine. Try using --debug for more information."
        exit 1
    fi
    
    echo -e "\nPerformance comparison completed."
    echo "Results saved to: performance_comparison.md"
    
    # Display a summary
    echo -e "\nSummary:"
    echo "Size : **$img_info**"
    echo
    echo '```'
    cat performance_comparison.md
    echo '```'
}

# Main function
main() {
    check_requirements
    
    # Check if we have enough arguments
    if [ $# -lt 2 ]; then
        usage
    fi
    
    ORIGINAL_IMG="$1"
    COMPRESSED_IMG="$2"
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
    DEBUG=false
    
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
            --debug)
                DEBUG=true
                shift
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
    
    # Create a temporary directory for test environments
    TEMP_DIR=$(mktemp -d)
    echo "Temporary directory: $TEMP_DIR"
    
    # Run the tests
    if [ "$FULL" = true ]; then
        # Test all tagged versions in a single comparison
        run_full_comparison "$ORIGINAL_IMG" "$COMPRESSED_IMG" "$WARMUP_RUNS" "$BENCHMARK_RUNS" "$TEMP_DIR" "$CURRENT_BRANCH" "$DEBUG"
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
        
        # Get image dimensions for the report
        img_info=$(python3 -c "from PIL import Image; img = Image.open('$ORIGINAL_IMG'); print(f'{img.width}x{img.height}')")
        
        echo "Image size: $img_info"
        
        # Run the benchmark
        hyperfine_cmd="hyperfine --warmup $WARMUP_RUNS --runs $BENCHMARK_RUNS"
        hyperfine_cmd="$hyperfine_cmd \"python3 -m ssimulacra2.cli '$ORIGINAL_IMG' '$COMPRESSED_IMG' --quiet\""
        hyperfine_cmd="$hyperfine_cmd --export-markdown \"performance_current.md\""
        
        echo "Command: $hyperfine_cmd"
        
        eval "$hyperfine_cmd"
        
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
    
    # Clean up
    if [ "$DEBUG" = true ]; then
        echo "Temporary directory preserved for debugging: $TEMP_DIR"
    else
        echo "Cleaning up temporary directory: $TEMP_DIR"
        rm -rf "$TEMP_DIR"
    fi
    
    # Return to the original branch
    git checkout "$CURRENT_BRANCH" 2>/dev/null
    
    echo "Performance benchmarking completed."
}

main "$@"
