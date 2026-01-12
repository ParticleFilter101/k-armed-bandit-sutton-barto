#!/bin/bash

# Ensure the script stops if a command fails
set -e

# Install requirements if needed (optional)
# pip install numpy matplotlib pyyaml tqdm

echo "Starting K-Armed Bandit Experiments..."

# Option 1: Run with defaults from config
# python main.py

# Option 2: Run with custom epsilons and custom output folder (as requested)
python main.py \
    --epsilons 1 0.1 0.01 0.0 \
    --output "./results/comparison_study"

echo "Experiment complete. Check ./results/comparison_study for plots."