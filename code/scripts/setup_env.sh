#!/bin/bash
# Script to set up a conda environment for the hive experiment

# Go to the project root directory
cd "$(dirname "$0")/.."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create a new conda environment called 'hive'
echo "Creating new conda environment 'hive'..."
conda create -y -n hive python=3.9

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate hive

# Install required packages
echo "Installing required packages..."
conda install -y numpy matplotlib

echo "\nConda environment 'hive' setup complete!\n"
echo "To use this environment, run:"
echo "  conda activate hive"
echo "\nAnd then run the experiment script with:"
echo "  ./scripts/run_experiment.sh --worlds 100 --steps 1000"
echo "\nWhen you're done, you can deactivate the environment with:"
echo "  conda deactivate"

