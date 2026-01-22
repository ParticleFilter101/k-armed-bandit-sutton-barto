import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.ModifiedNonStationaryBandit import NonStationaryMAB

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment_for_method(method_name: str, step_size: float, epsilon: float, config: dict):
    """
    Runs the full experiment for a single method.
    
    Args:
        method_name: Name of the method (for display)
        step_size: None for sample average, or constant value (e.g., 0.1)
        epsilon: Exploration rate
        config: Configuration dictionary
    """
    num_runs = config['experiment']['num_runs']
    num_steps = config['nonstationary']['num_steps']  # Use longer steps for non-stationary
    num_arms = config['experiment']['num_arms']

    rewards = np.zeros((num_runs, num_steps))
    optimal_counts = np.zeros((num_runs, num_steps))

    env = NonStationaryMAB(num_arms=num_arms, epsilon=epsilon, step_size=step_size)

    print(f"Running experiment for {method_name}")
    for run in tqdm(range(num_runs), desc=method_name):
        env.reset()
        
        for step in range(num_steps):
            reward, is_optimal = env.step()
            rewards[run, step] = reward
            optimal_counts[run, step] = 1 if is_optimal else 0

    # Average over runs
    avg_rewards = np.mean(rewards, axis=0)
    avg_optimal_pct = np.mean(optimal_counts, axis=0) * 100

    return avg_rewards, avg_optimal_pct

def plot_results(results: dict, output_dir: str, num_runs: int, num_steps: int):
    """Plots and saves results for multiple methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Average Reward Plot
    plt.figure(figsize=(12, 6))
    for method_name, (avg_rewards, _) in results.items():
        plt.plot(avg_rewards, label=method_name, linewidth=1.5)
    
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title(f"Non-Stationary Bandit: Average Reward over {num_runs} Runs", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nonstationary_average_rewards.png"), dpi=300)
    plt.close()

    # 2. Optimal Action Percentage Plot
    plt.figure(figsize=(12, 6))
    for method_name, (_, avg_optimal) in results.items():
        plt.plot(avg_optimal, label=method_name, linewidth=1.5)
    
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("% Optimal Action", fontsize=12)
    plt.title(f"Non-Stationary Bandit: Optimal Action Selection over {num_runs} Runs", fontsize=14)
    plt.ylim(0, 100)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nonstationary_optimal_percentage.png"), dpi=300)
    plt.close()

    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Non-Stationary K-Armed Bandit Experiment")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, help='Output directory (overrides config)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Constant step-size alpha (default: 0.1)')
    args = parser.parse_args()

    config = load_config(args.config)

    # Determine output directory
    output_dir = args.output if args.output else config['nonstationary']['output_dir']

    all_results = {}

    # Method 1: Sample Averages (step_size = None means 1/n)
    print("\n" + "="*60)
    print("Method 1: Sample Averages (α = 1/n)")
    print("="*60)
    avg_rewards_sample, avg_optimal_sample = run_experiment_for_method(
        method_name=f"Sample Average (ε={args.epsilon})",
        step_size=None,  # This triggers sample average
        epsilon=args.epsilon,
        config=config
    )
    all_results[f"Sample Average (ε={args.epsilon})"] = (avg_rewards_sample, avg_optimal_sample)

    # Method 2: Constant Step-Size
    print("\n" + "="*60)
    print(f"Method 2: Constant Step-Size (α = {args.alpha})")
    print("="*60)
    avg_rewards_constant, avg_optimal_constant = run_experiment_for_method(
        method_name=f"Constant α={args.alpha} (ε={args.epsilon})",
        step_size=args.alpha,
        epsilon=args.epsilon,
        config=config
    )
    all_results[f"Constant α={args.alpha} (ε={args.epsilon})"] = (avg_rewards_constant, avg_optimal_constant)

    # Plot combined results
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    plot_results(all_results, output_dir, config['experiment']['num_runs'], 
                 config['nonstationary']['num_steps'])
    
    print("\n✓ Experiment complete!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
