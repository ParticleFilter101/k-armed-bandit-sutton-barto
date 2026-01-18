# K-Armed Bandit Testbed

A Python implementation of the **K-Armed Bandit** problem from Sutton & Barto's *Reinforcement Learning: An Introduction*. This repository provides a modular, configurable testbed for experimenting with $\epsilon$-greedy agents.

## Project Structure
```text
k_armed_bandit/
├── config.yaml             # Hyperparameters configuration
├── main.py                 # Experiment runner & plotter
├── run_experiments.sh      # One-click execution script
└── src/
    ├── __init__.py         # Package exporter
    └── bandit.py           # Core Environment Logic