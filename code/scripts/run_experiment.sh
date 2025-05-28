#!/bin/bash
# Wrapper script for running the random policy experiment

# Function to display help message
show_help() {
    echo "Random Policy Experiment for Hive Simulation"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --worlds N         Number of parallel worlds to simulate (= number of episodes) [default: 100]"
    echo "  --steps N          Number of steps per episode [default: 1000]"
    echo "  --random-seed N    Random seed for reproducibility [default: 42]"
    echo "  --exec-mode MODE   Execution mode (CPU or CUDA) [default: CPU]"
    echo "  --help             Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --worlds 100 --steps 1000                  # Run 100 episodes with 1000 steps each"
    echo "  $0 --worlds 10 --steps 500 --random-seed 123  # Run 10 episodes with 500 steps using seed 123"
    echo "  $0 --exec-mode CUDA                          # Run with CUDA if available"
    echo ""
    exit 0
}

# Check for help flag
for arg in "$@"; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
        show_help
    fi
done

# Make sure we're in the right directory
cd "$(dirname "$0")/.."

# Ensure the build directory exists and project is built
if [ ! -d "build" ] || [ ! -f "build/headless" ]; then
    echo "Build directory or executable not found. Building project..."
    cmake -B build
    cmake --build build
fi

# Create results directory if it doesn't exist
mkdir -p results

# Run the experiment
python3 scripts/run_random_policy_experiment.py "$@"

# If successful, suggest viewing the results
if [ $? -eq 0 ]; then
    echo "\nResults have been saved to the 'results' directory."
    echo "Check the PNG files for visualizations of the experiment results."
fi
