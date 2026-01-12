import numpy as np
from typing import Tuple, Dict, List

class MabTestBed:
    """
    Multi-Armed Bandit test bed for epsilon-greedy algorithm.
    Ref: Sutton & Barto, Reinforcement Learning: An Introduction.
    """

    def __init__(self, num_arms: int, epsilon: float):
        """
        Args:
            num_arms: Number of arms in the bandit.
            epsilon: Exploration probability.
        """
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.q_true = np.zeros(num_arms)       # True optimal value of arms
        self.q_estimates = np.zeros(num_arms)  # Estimated value of arms
        self.action_counts = np.zeros(num_arms) # Number of times arm picked

    def reset(self):
        """
        Resets the bandit for a new independent run.
        Generates new true values from N(0,1) and resets estimates.
        """
        self.q_true = np.random.normal(loc=0.0, scale=1.0, size=self.num_arms)
        self.q_estimates = np.zeros(self.num_arms)
        self.action_counts = np.zeros(self.num_arms)

    def _get_best_arm(self) -> int:
        """Returns the index of the arm with the highest current estimate."""
        # We break ties randomly, which np.argmax does not do by default
        max_val = np.max(self.q_estimates)
        candidates = np.flatnonzero(self.q_estimates == max_val)
        return np.random.choice(candidates)

    def step(self) -> Tuple[float, bool]:
        """
        Takes one step (action) in the environment.
        
        Returns:
            reward: The reward received.
            is_optimal: Boolean, true if the optimal arm was chosen.
        """
        # 1. Choose Action (Epsilon-Greedy)
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_arms)
        else:
            action = self._get_best_arm()

        # 2. Get Reward (True Value + Noise)
        reward = np.random.normal(loc=self.q_true[action], scale=1.0)

        # 3. Update Estimates (Incremental Implementation)
        self.action_counts[action] += 1
        step_size = 1.0 / self.action_counts[action]
        self.q_estimates[action] += step_size * (reward - self.q_estimates[action])

        # 4. Check optimality (compare against the true best arm)
        true_best_arm = np.argmax(self.q_true)
        is_optimal = (action == true_best_arm)

        return reward, is_optimal