import config
import numpy as np

class Algorithm:
    """Base class for optimization algorithms"""
    def __init__(self, data_samples, conformal_predictors, rewards) -> None:
        self.data = data_samples
        self.rewards = rewards
        self.conf_predictors = conformal_predictors

        # Initialize current and max timestep
        self.current_time_step = 0
        self.T = len(self.data)
        
        # Pulled arms per time step
        self.arm_t = np.zeros(self.T)

        # Initialize counters
        # Cumulative reward per alpha
        self.cumulative_reward = {alpha : 0 for alpha in self.conf_predictors.alpha_values}
        # Number of observed rewards 
        self.n_rewards = {alpha : 0 for alpha in self.conf_predictors.alpha_values}

    def sample_reward(self, x, s):
        if s == 0:
            return 0
        index_q = self.rewards.index == x
        set_q = self.rewards.set == s    
        reward_df = self.rewards[(index_q)&(set_q)].reward.sample(random_state=config.numpy_rng)
        return reward_df.values[0]

    def update_counters(self, alpha, observed_reward):
        """Update counters for an alpha value"""
        self.cumulative_reward[alpha]+=observed_reward
        self.n_rewards[alpha]+=1

    def average_reward(self, alpha):
        return self.cumulative_reward[alpha]/self.n_rewards[alpha]

    def error(self, alpha):
        pass

    def run_algorithm(self):
        pass
