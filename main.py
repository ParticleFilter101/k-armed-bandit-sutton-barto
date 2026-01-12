import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Professional progress bar
from src import MabTestBed

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment_for_epsilon(epsilon: float, config: dict):
    """Runs the full experiment for a single epsilon value."""
    num_runs = config['experiment']['num_runs']
    num_steps = config['experiment']['num_steps']
    num_arms = config['experiment']['num_arms']

    rewards = np.zeros((num_runs, num_steps))
    optimal_counts = np.zeros((num_runs, num_steps))

    env = MabTestBed(num_arms=num_arms, epsilon=epsilon)

    # Use tqdm for a progress bar in the terminal
    print(f"Running experiment for Epsilon = {epsilon}")
    for run in tqdm(range(num_runs), desc=f"ε={epsilon}"):
        env.reset() # Generate new arm values for this run
        
        for step in range(num_steps):
            reward, is_optimal = env.step()
            rewards[run, step] = reward
            optimal_counts[run, step] = 1 if is_optimal else 0

    # Average over runs
    avg_rewards = np.mean(rewards, axis=0)
    avg_optimal_pct = np.mean(optimal_counts, axis=0) * 100

    return avg_rewards, avg_optimal_pct

def plot_results(results: dict, output_dir: str, num_runs: int):
    """Plots and saves results for multiple epsilons."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Average Reward Plot
    plt.figure(figsize=(10, 6))
    for eps, (avg_rewards, _) in results.items():
        plt.plot(avg_rewards, label=f'$\epsilon = {eps}$')
    
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward over {num_runs} Runs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "average_rewards.png"), dpi=300)
    plt.close()

    # 2. Optimal Action Percentage Plot
    plt.figure(figsize=(10, 6))
    for eps, (_, avg_optimal) in results.items():
        plt.plot(avg_optimal, label=f'$\epsilon = {eps}$')
    
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title(f"Optimal Action Selection over {num_runs} Runs")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "optimal_percentage.png"), dpi=300)
    plt.close()

    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="K-Armed Bandit Experiment Runner")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    # Allow user to pass multiple epsilons via command line
    parser.add_argument('--epsilons', nargs='+', type=float, help='List of epsilon values to test (overrides config)')
    parser.add_argument('--output', type=str, help='Output directory (overrides config)')
    args = parser.parse_args()

    config = load_config(args.config)

    # Determine epsilons to run
    eps_list = args.epsilons if args.epsilons else config['default_epsilons']
    
    # Determine output directory
    output_dir = args.output if args.output else config['experiment']['output_dir']

    all_results = {}

    # Run experiments
    for eps in eps_list:
        avg_rewards, avg_optimal_pct = run_experiment_for_epsilon(eps, config)
        all_results[eps] = (avg_rewards, avg_optimal_pct)

    # Plot combined results
    plot_results(all_results, output_dir, config['experiment']['num_runs'])

if __name__ == "__main__":
    main()