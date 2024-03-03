from algorithms.algorithm import Algorithm
import numpy as np
from algorithms.utils import *

class Test(Algorithm):
    """Empirical success probability for each alpha value"""
    def __init__(self, data_samples, conformal_predictors, rewards) -> None:
        super().__init__(data_samples, conformal_predictors, rewards)
        self.average_rewards = np.vectorize(self.average_reward)
        self.std_errors = np.vectorize(self.std_error)

    def standard_deviation(self, alpha):
        all_rewards = np.zeros(self.n_rewards[alpha])
        for i in range(self.cumulative_reward[alpha]):
            all_rewards[i] = 1

        return np.std(all_rewards)

    def std_error(self, alpha):
        std = self.standard_deviation(alpha)
        return std / np.sqrt(self.n_rewards[alpha])
        
    def run_algorithm(self, regret=False):
        while self.current_time_step<self.T:
            data_sample = self.data[self.current_time_step]
            set_to_alpha, _ = self.conf_predictors.set_sizes_alphas(data_sample)
            rew_slice = self.rewards.loc[data_sample][['set', 'reward']]
            human_alone_rew_values = rew_slice[rew_slice['set'] == 16].reward.values
            for s in set_to_alpha:
                rew_values = rew_slice[rew_slice['set'] == s].reward.values
                for a in set_to_alpha[s]:
                    if not s:
                        if not regret:
                            # Deployment mode, if the set if empty human picks on their own
                            for reward in human_alone_rew_values:
                                assert type(reward) == np.bool_ or type(reward) == bool or type(reward)==int
                                self.update_counters(a, int(reward))
                        else:
                            # Optimization mode, if the set is empty the reward is 0
                            for _ in range(5):
                                self.update_counters(a, 0)
                    else:
                        for reward in rew_values:
                            assert type(reward) == np.bool_ or type(reward) == bool or type(reward)==int
                            self.update_counters(a, int(reward))
            self.current_time_step+=1
        
        average_acc_per_alpha = self.average_rewards(self.conf_predictors.alpha_values)
        se_acc_per_alpha = self.std_errors(self.conf_predictors.alpha_values)
        
        return average_acc_per_alpha, se_acc_per_alpha
