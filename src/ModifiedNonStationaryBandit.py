import numpy as np
from typing import Tuple

class NonStationaryMAB:
    """
    Non-stationary Multi-Armed Bandit where true action values perform random walks.
    
    Unlike the stationary version, the true q values:
    - Start equal (typically at 0)
    - Change at each step via random walk: q_true += N(0, 0.01)
    
    This implementation supports both:
    - Sample averages (step_size = 1/n)
    - Constant step-size (fixed alpha)
    """

    def __init__(self, num_arms: int, epsilon: float, step_size: float = None):
        """
        Args:
            num_arms: Number of arms in the bandit.
            epsilon: Exploration probability.
            step_size: If None, uses sample averages (1/n). 
                      If provided (e.g., 0.1), uses constant step-size.
        """
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.step_size = step_size  # None means sample average, else constant
        self.q_true = np.zeros(num_arms)       # True optimal value of arms
        self.q_estimates = np.zeros(num_arms)  # Estimated value of arms
        self.action_counts = np.zeros(num_arms) # Number of times arm picked

    def reset(self):
        """
        Resets the bandit for a new independent run.
        All true values start equal at 0 (for non-stationary setting).
        """
        self.q_true = np.zeros(self.num_arms)  # Start all equal at 0
        self.q_estimates = np.zeros(self.num_arms)
        self.action_counts = np.zeros(self.num_arms)

    def _get_best_arm(self) -> int:
        """Returns the index of the arm with the highest current estimate."""
        # Break ties randomly
        max_val = np.max(self.q_estimates)
        candidates = np.flatnonzero(self.q_estimates == max_val)
        return np.random.choice(candidates)

    def step(self) -> Tuple[float, bool]:
        """
        Takes one step (action) in the non-stationary environment.
        
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

        # 3. Update Estimates
        self.action_counts[action] += 1
        if self.step_size is None:
            # Sample average: step_size = 1/n
            alpha = 1.0 / self.action_counts[action]
        else:
            # Constant step-size
            alpha = self.step_size
        
        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])

        # 4. Check optimality (compare against the true best arm)
        true_best_arm = np.argmax(self.q_true)
        is_optimal = (action == true_best_arm)

        # 5. RANDOM WALK: Update all true q values for non-stationarity
        self.q_true += np.random.normal(loc=0.0, scale=0.01, size=self.num_arms)

        return reward, is_optimal
