import config 
from algorithms.algorithm import Algorithm
import numpy as np
from algorithms.utils import *
from models.preprocess_predictions import get_idx_min_valid_non_singleton_set as pivot_sets
from collections import defaultdict

class UCB(Algorithm):
    """Vanilla UCB1"""
    def __init__(self, data_samples, conformal_predictors, rewards) -> None:
        super().__init__(data_samples, conformal_predictors, rewards)
        self.ucbs = np.vectorize(self.ucb)
        print("Algorithm initialization done")
        print(f"t = {self.current_time_step}")

    def error(self, alpha):
        return np.sqrt(2*np.log(self.T) / self.n_rewards[alpha])

    def ucb(self, alpha):
        if self.n_rewards[alpha]>0:
            return self.average_reward(alpha) + self.error(alpha)
        else:
            return 0
    
    def update_rewards(self, alpha_to_deploy):
        """Updates the counters with the revealed rewards"""
        # Update pulled arms
        self.arm_t[self.current_time_step] = alpha_to_deploy
        
        print("Updating rewards")
        data_sample = self.data[self.current_time_step]

        set_to_alpha, alpha_to_set = self.conf_predictors.set_sizes_alphas(data_sample)
      
        deployed_set = alpha_to_set[alpha_to_deploy]
        observed_reward = self.sample_reward(data_sample, deployed_set)
        print(f"Data sample: {data_sample}")
        print(f"Deployed set {deployed_set}")
        print(f"Observed reward {observed_reward}")
        assert type(observed_reward) == np.bool_ or type(observed_reward) == bool or type(observed_reward)==int
        
        self.update_counters(alpha_to_deploy, int(observed_reward))
        self.current_time_step+=1
     
    def collect_rewards_for_each_arm(self):
        for alpha in self.conf_predictors.alpha_values:
            self.update_rewards(alpha)

    def run_algorithm(self):

        self.collect_rewards_for_each_arm()

        while self.current_time_step < self.T: 
            # Deploy arm with the highest ucb
            alpha_to_deploy_idx = self.ucbs(self.conf_predictors.alpha_values).argmax()
            alpha_to_deploy = self.conf_predictors.alpha_values[alpha_to_deploy_idx]
            self.update_rewards(alpha_to_deploy)
        
        # Save rewards per time step for this run
        path = f"{config.ROOT_DIR}/{config.results_path}/regret/{config.algorithm_key}/seed_{config.run_no_seed}_cal_{config.run_no_cal}.npy"
        np.save(path, self.arm_t)
        return alpha_to_deploy,alpha_to_deploy_idx
  
class UCBOurs(Algorithm):
    """Counterfactual UCB1"""
    def __init__(self, data_samples, conformal_predictors, rewards) -> None:
        super().__init__(data_samples, conformal_predictors, rewards)
        self.ucbs = np.vectorize(self.ucb)
        self.pivot_sets = pivot_sets().to_dict()
        print("Algorithm initialization done")
        print(f"t = {self.current_time_step}")
        self.flag_new_sample = True

    def error(self, alpha):
        return np.sqrt(2*np.log(self.T) / self.n_rewards[alpha])

    def ucb(self, alpha):
        if self.n_rewards[alpha]>0:
            return self.average_reward(alpha) + self.error(alpha)
        else:
            return 0
    
    def update_rewards_init(self, alpha_to_deploy):
        """Updates the counters with the revealed rewards and returns
           where to search next"""
        # Update pulled arms
        self.arm_t[self.current_time_step] = alpha_to_deploy
        
        print("Updating rewards")
        data_sample = self.data[self.current_time_step]

        set_to_alpha, alpha_to_set = self.conf_predictors.set_sizes_alphas(data_sample)
        
        deployed_set = alpha_to_set[alpha_to_deploy]
        observed_reward = self.sample_reward(data_sample, deployed_set)
        assert type(observed_reward) == np.bool_ or type(observed_reward) == bool or type(observed_reward)==int
        # Correction as pivot set is an index (0 to 15)
        pivot_set = self.pivot_sets[data_sample] + 1
        
        if observed_reward:
            for s in range(0,deployed_set+1):
                if s < pivot_set:
                    if s in set_to_alpha:
                        for a in set_to_alpha[s]:
                            self.update_counters(a, 0)
                else:
                    if s in set_to_alpha:
                        for a in set_to_alpha[s]:
                            self.update_counters(a, 1)
                next_search, alpha_bound = Search.UPPER, min(set_to_alpha[deployed_set])
        else:
            for s in range(0,pivot_set):
                if s in set_to_alpha:
                    for a in set_to_alpha[s]:
                        self.update_counters(a, 0)
            if deployed_set < pivot_set:
                # Pull arms after pivot
                sets = list(set_to_alpha.keys())
                before_pivot_idx = np.searchsorted(sets, pivot_set-1)
                before_pivot_idx = min(before_pivot_idx, len(set_to_alpha) - 1)
                before_pivot = sets[before_pivot_idx]
                next_search, alpha_bound = Search.UPPER, min(set_to_alpha[before_pivot])
            else:
                for s in range(deployed_set, config.N_LABELS+1):
                    if s in set_to_alpha:
                        for a in set_to_alpha[s]:
                            self.update_counters(a, 0)         
                next_search, alpha_bound = Search.LOWER, max(set_to_alpha[deployed_set])
        self.current_time_step+=1
        return next_search, alpha_bound
 
    def collect_rewards_for_each_alpha(self, status, alphas_to_search):
        if status == Status.SEARCH:
            alpha_to_deploy = np.nanpercentile(alphas_to_search, 50, method='lower') if config.numpy_rng.integers(0,1,endpoint=True) else np.nanpercentile(alphas_to_search, 50, method='higher')
            print(f"Pulling arm {alpha_to_deploy}")
            next_search, bound_alpha = self.update_rewards_init(alpha_to_deploy)
            print(f"t = {self.current_time_step}")
         
            if next_search == Search.UPPER:
                alphas_to_search_next = alphas_to_search[alphas_to_search < bound_alpha]
                
            elif next_search == Search.LOWER:
                alphas_to_search_next = alphas_to_search[alphas_to_search > bound_alpha]
                
            else:
                alphas_to_search_next = []

            if (not len(alphas_to_search_next)) or (self.current_time_step==self.T):
                return self.collect_rewards_for_each_alpha(Status.DONE, [])
            self.collect_rewards_for_each_alpha(Status.SEARCH, alphas_to_search_next)
        else:
            return
        return

    def update_rewards(self, alpha_to_deploy):
        # Update pulled arms
        self.arm_t[self.current_time_step] = alpha_to_deploy
        
        print("Updating rewards")
        data_sample = self.data[self.current_time_step]

        set_to_alpha, alpha_to_set = self.conf_predictors.set_sizes_alphas(data_sample)

        deployed_set = alpha_to_set[alpha_to_deploy]
        observed_reward = self.sample_reward(data_sample, deployed_set)
        
        print(f"Data sample: {data_sample}")
        print(f"Deployed set {deployed_set}")
        print(f"Observed reward {observed_reward}")
        assert type(observed_reward) == np.bool_ or type(observed_reward) == bool or type(observed_reward)==int
        # Correction as pivot set is an index (0 to 15)
        pivot_set = self.pivot_sets[data_sample] + 1
        print(f"Pivot set {pivot_set}")
        if deployed_set >= pivot_set:
            for s in range(0, pivot_set):
                if s in set_to_alpha:
                    for a in set_to_alpha[s]:
                        self.update_counters(a, 0)
            if observed_reward:
                for s in range(pivot_set, deployed_set+1):
                    if s in set_to_alpha:    
                        for a in set_to_alpha[s]:
                            self.update_counters(a, 1)
            else:
                for s in range(deployed_set, config.N_LABELS+1):
                    if s in set_to_alpha:
                        for a in set_to_alpha[s]:
                            self.update_counters(a, 0)
        else:
            for s in range(0, pivot_set):
                if s in set_to_alpha:
                    for a in set_to_alpha[s]:
                        self.update_counters(a, 0)
        self.current_time_step+=1

    def run_algorithm(self):

        self.collect_rewards_for_each_alpha(Status.SEARCH, self.conf_predictors.alpha_values)
        
        while self.current_time_step < self.T: 
            # Deploy arm with the highest ucb
            alpha_to_deploy_idx = self.ucbs(self.conf_predictors.alpha_values).argmax()
            alpha_to_deploy = self.conf_predictors.alpha_values[alpha_to_deploy_idx]
            self.update_rewards(alpha_to_deploy)
        # Save arms per timestep
        path = f"{config.ROOT_DIR}/{config.results_path}/regret/{config.algorithm_key}/seed_{config.run_no_seed}_cal_{config.run_no_cal}.npy"
        np.save(path, self.arm_t)
        return alpha_to_deploy,alpha_to_deploy_idx

class UCBNoMonotonicity(Algorithm):
    """Assumption-free counterfactual UCB1"""
    def __init__(self, data_samples, conformal_predictors, rewards) -> None:
        super().__init__(data_samples, conformal_predictors, rewards)
        self.ucbs = np.vectorize(self.ucb)
        self.pivot_sets = pivot_sets().to_dict()
        print("Algorithm initialization done")
        print(f"t = {self.current_time_step}")
        self.is_alpha_updated = defaultdict(lambda:0)

    def error(self, alpha):
        return np.sqrt(2*np.log(self.T) / self.n_rewards[alpha])

    def ucb(self, alpha):
        if self.n_rewards[alpha]>0:
            return self.average_reward(alpha) + self.error(alpha)
        else:
            return 0
    
    def update_rewards_init(self, alpha_to_deploy):
        """Updates the counters with the revealed rewards and returns
           where to search next"""
        data_sample = self.data[self.current_time_step]

        set_to_alpha, alpha_to_set = self.conf_predictors.set_sizes_alphas(data_sample)
      
        deployed_set = alpha_to_set[alpha_to_deploy]
        observed_reward = self.sample_reward(data_sample, deployed_set)
        assert type(observed_reward) == np.bool_ or type(observed_reward) == bool or type(observed_reward)==int
        
        pivot_set = self.pivot_sets[data_sample] + 1
        for s in range(0,pivot_set):
            if s in set_to_alpha:
                for a in set_to_alpha[s]:
                    self.update_counters(a, 0)
                    self.is_alpha_updated[a] = 1
        
        for a in set_to_alpha[deployed_set]:
            if (not np.isnan(a)) and (not self.is_alpha_updated[a]) :
                self.update_counters(a, int(observed_reward))
                self.is_alpha_updated[a] = 1
        
        self.arm_t[self.current_time_step] = alpha_to_deploy    
        self.current_time_step+=1
        
    def collect_rewards_for_each_alpha(self, alphas_to_search):
        for alpha in alphas_to_search:
            if not np.isnan(alpha):
                if not self.is_alpha_updated[alpha]:
                    self.update_rewards_init(alpha)
            if self.current_time_step == self.T:
                return

    def update_rewards(self, alpha_to_deploy):
        # Update pulled arms
        self.arm_t[self.current_time_step] = alpha_to_deploy
        
        print("Updating rewards")
        data_sample = self.data[self.current_time_step]

        set_to_alpha, alpha_to_set = self.conf_predictors.set_sizes_alphas(data_sample)
       
        deployed_set = alpha_to_set[alpha_to_deploy]
        observed_reward = self.sample_reward(data_sample, deployed_set)

        print(f"Data sample: {data_sample}")
        print(f"Deployed set {deployed_set}")
        print(f"Observed reward {observed_reward}")
        assert type(observed_reward) == np.bool_ or type(observed_reward) == bool or type(observed_reward)==int
        # Correction as pivot set is an index (0 to 15)
        pivot_set = self.pivot_sets[data_sample] + 1
        print(f"Pivot set {pivot_set}")
        if deployed_set >= pivot_set:
            for s in range(0, pivot_set):
                if s in set_to_alpha:
                    for a in set_to_alpha[s]:
                        self.update_counters(a, 0)
            if observed_reward:
                for a in set_to_alpha[deployed_set]:
                    self.update_counters(a, 1)
            else:
                for a in set_to_alpha[deployed_set]:
                    self.update_counters(a, 0)
        else:
            for s in range(0, pivot_set):
                if s in set_to_alpha:
                    for a in set_to_alpha[s]:
                        self.update_counters(a, 0)
        self.current_time_step+=1

    def run_algorithm(self):

        self.collect_rewards_for_each_alpha(self.conf_predictors.alpha_values)
        
        while self.current_time_step < self.T: 
            # Deploy arm with the highest ucb
            alpha_to_deploy_idx = self.ucbs(self.conf_predictors.alpha_values).argmax()
            alpha_to_deploy = self.conf_predictors.alpha_values[alpha_to_deploy_idx]
            self.update_rewards(alpha_to_deploy)
        # Save arms for each timestep
        path = f"{config.ROOT_DIR}/{config.results_path}/regret/{config.algorithm_key}/seed_{config.run_no_seed}_cal_{config.run_no_cal}.npy"
        np.save(path, self.arm_t)
        
        return alpha_to_deploy,alpha_to_deploy_idx
