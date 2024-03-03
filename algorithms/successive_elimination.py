from algorithms.algorithm import Algorithm
import numpy as np
import config
from models.preprocess_predictions import get_idx_min_valid_non_singleton_set as pivot_sets
import numpy.ma as ma
from collections import defaultdict
from algorithms.utils import *
import json

class SuccessiveElimination(Algorithm):
    """Vanilla successive elimination algorithm"""
    def __init__(self, data_samples, conformal_predictors, rewards) -> None:
        super().__init__(data_samples, conformal_predictors, rewards)
        self.ucbs = np.vectorize(self.ucb)
        self.lcbs = np.vectorize(self.lcb)
  
    def error(self, alpha):
        return np.sqrt(2*np.log(self.T) / self.n_rewards[alpha])

    def ucb(self, alpha):
        if self.n_rewards[alpha]>0:
            return self.average_reward(alpha) + self.error(alpha)
        else:
            return 0
    
    def lcb(self, alpha):
        if self.n_rewards[alpha]>0:
            return self.average_reward(alpha) - self.error(alpha)
        else:
            return 0

    def update_rewards(self, alpha_to_deploy):
        """Updates the counters with the revealed rewards"""
        # Update pulled arms
        self.arm_t[self.current_time_step] = alpha_to_deploy
        data_sample = self.data[self.current_time_step]

        set_to_alpha, alpha_to_set = self.conf_predictors.set_sizes_alphas(data_sample)
      
        deployed_set = alpha_to_set[alpha_to_deploy]
        observed_reward = self.sample_reward(data_sample, deployed_set)
        assert type(observed_reward) == np.bool_ or type(observed_reward) == bool or type(observed_reward)==int
        
        self.update_counters(alpha_to_deploy, int(observed_reward))
        self.current_time_step+=1
        
    def collect_rewards_for_each_alpha(self, alphas_to_search):
        for alpha in alphas_to_search:
            if not np.isnan(alpha):
                self.update_rewards(alpha)
                if self.current_time_step == self.T:
                    return

    def deactivate_rule(self, active_alphas_mask):
        alphas_ucbs = self.ucbs(self.conf_predictors.alpha_values)
        max_lcb = self.lcbs(self.conf_predictors.alpha_values).max()
      
        for i,_ in enumerate(active_alphas_mask):
            if not (alphas_ucbs[i] >= max_lcb):
                active_alphas_mask[i] = 1
        assert len(active_alphas_mask) == len(self.conf_predictors.alpha_values)
        return active_alphas_mask

    def run_algorithm(self):
        active_alphas_mask = np.zeros_like(self.conf_predictors.alpha_values).astype(bool)
        masked_alphas = ma.masked_array(self.conf_predictors.alpha_values, mask=active_alphas_mask)

        while self.current_time_step < self.T and active_alphas_mask.sum() < (len(active_alphas_mask)-1):
            masked_alphas = ma.masked_array(self.conf_predictors.alpha_values, mask=active_alphas_mask)
            masked_alphas = ma.filled(masked_alphas, np.nan)
            self.collect_rewards_for_each_alpha(masked_alphas)
            active_alphas_mask = self.deactivate_rule(active_alphas_mask)
        
        print(f"{len(active_alphas_mask) - active_alphas_mask.sum()} active arms")

        # If many alpha values remain active return the one with the highest lcb
        best_alpha_idx = self.lcbs(self.conf_predictors.alpha_values).argmax()
        best_alpha_found = self.conf_predictors.alpha_values[best_alpha_idx]
        
        masked_alphas = ma.masked_array(self.conf_predictors.alpha_values, mask=active_alphas_mask)
        print(masked_alphas)
        avs = []
        n_rs = []
        for a in masked_alphas.compressed():      
            avs.append(self.average_reward(a))
            n_rs.append(self.n_rewards[a])
        print(avs)
        print(n_rs)
        
        # Save arms per time step for this run
        path = f"{config.ROOT_DIR}/{config.results_path}/regret/{config.algorithm_key}/seed_{config.run_no_seed}_cal_{config.run_no_cal}.npy"
        np.save(path, self.arm_t)

        return best_alpha_found,best_alpha_idx
 
class SuccessiveEliminationOurs(Algorithm):
    """Counterfactual successive elimination algorithm"""
    def __init__(self, data_samples, conformal_predictors, rewards) -> None:
        super().__init__(data_samples, conformal_predictors, rewards)
        self.ucbs = np.vectorize(self.ucb)
        self.lcbs = np.vectorize(self.lcb)
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
    
    def lcb(self, alpha):
        if self.n_rewards[alpha]>0:
            return self.average_reward(alpha) - self.error(alpha)
        else:
            return 0

    def update_rewards(self, alpha_to_deploy):
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
            next_search, bound_alpha = self.update_rewards(alpha_to_deploy)
            print(f"t = {self.current_time_step}")
         
            if next_search == Search.UPPER:
                alphas_to_search_next = alphas_to_search[alphas_to_search < bound_alpha]
                
            elif next_search == Search.LOWER:
                alphas_to_search_next = alphas_to_search[alphas_to_search > bound_alpha]
            else:
                alphas_to_search_next = []

            if (not len(alphas_to_search_next)) or (self.current_time_step==self.T):
                print('Done')
                return self.collect_rewards_for_each_alpha(Status.DONE, [])
            self.collect_rewards_for_each_alpha(Status.SEARCH, alphas_to_search_next)
        else:
            return
        return

    def deactivate_rule(self, active_alphas_mask):
        alphas_ucbs = self.ucbs(self.conf_predictors.alpha_values)
        max_lcb = self.lcbs(self.conf_predictors.alpha_values).max()

        for i,_ in enumerate(active_alphas_mask):
            if not (alphas_ucbs[i] >= max_lcb):
                active_alphas_mask[i] = 1

        return active_alphas_mask

    def run_algorithm(self):
        active_alphas_mask = np.zeros_like(self.conf_predictors.alpha_values).astype(bool)
        masked_alphas = ma.masked_array(self.conf_predictors.alpha_values, mask=active_alphas_mask)
        while self.current_time_step < self.T and active_alphas_mask.sum() < (len(active_alphas_mask)-1):
            masked_alphas = ma.masked_array(self.conf_predictors.alpha_values, mask=active_alphas_mask)
            masked_alphas = ma.filled(masked_alphas, np.nan)
            self.collect_rewards_for_each_alpha(Status.SEARCH, masked_alphas)
            active_alphas_mask = self.deactivate_rule(active_alphas_mask)
        
        print(f"{len(active_alphas_mask) - active_alphas_mask.sum()} active arms")

        # If many alpha values remain active return the one with the highest lcb
        best_alpha_idx = self.lcbs(self.conf_predictors.alpha_values).argmax()
        best_alpha_found = self.conf_predictors.alpha_values[best_alpha_idx]
        
        masked_alphas = ma.masked_array(self.conf_predictors.alpha_values, mask=active_alphas_mask)
        avs = []
        n_rs = []
        active_arms = []
        for a in masked_alphas.compressed():
            active_arms.append(a)
            avs.append(self.average_reward(a))
            n_rs.append(self.n_rewards[a])
        print(avs)
        print(n_rs)
        print(f"Index of arm with highest lcb {self.lcbs(self.conf_predictors.alpha_values).argmax()}")
        # Save arms per time step
        path = f"{config.ROOT_DIR}/{config.results_path}/regret/{config.algorithm_key}/seed_{config.run_no_seed}_cal_{config.run_no_cal}.npy"
        np.save(path, self.arm_t)
        # Save active arms at the last round
        path = f"{config.ROOT_DIR}/{config.output_path}/{config.algorithm_key}/active_arms_run{config.run_no_seed}_calrun{config.run_no_cal}.json"
        with open(path, 'wt') as f:
            json.dump(active_arms, f)
        return best_alpha_found,best_alpha_idx

class SuccessiveEliminationNoMonotonicity(Algorithm):
    """Assumption-free counterfactual successive elimination algorithm"""
    def __init__(self, data_samples, conformal_predictors, rewards) -> None:
        super().__init__(data_samples, conformal_predictors, rewards)
        self.ucbs = np.vectorize(self.ucb)
        self.lcbs = np.vectorize(self.lcb)
        self.is_alpha_updated = defaultdict(lambda:0)
        self.pivot_sets = pivot_sets().to_dict()

    def error(self, alpha):
        return np.sqrt(2*np.log(self.T) / self.n_rewards[alpha])

    def ucb(self, alpha):
        if self.n_rewards[alpha]>0:
            return self.average_reward(alpha) + self.error(alpha)
        else:
            return 0
    
    def lcb(self, alpha):
        if self.n_rewards[alpha]>0:
            return self.average_reward(alpha) - self.error(alpha)
        else:
            return 0

    def update_rewards(self, alpha_to_deploy):
        """Updates the counters with the revealed rewards"""
        data_sample = self.data[self.current_time_step]

        set_to_alpha, alpha_to_set = self.conf_predictors.set_sizes_alphas(data_sample)
      
        deployed_set = alpha_to_set[alpha_to_deploy]
        observed_reward = self.sample_reward(data_sample, deployed_set)
        assert type(observed_reward) == np.bool_ or type(observed_reward) == bool or type(observed_reward)==int
        # Correction as pivot set is an index (0 to 15) 
        pivot_set = self.pivot_sets[data_sample] + 1

        for s in range(0,pivot_set):
            if s in set_to_alpha:
                for a in set_to_alpha[s]:
                    if not self.is_alpha_updated[a]:
                        self.update_counters(a, 0)
                        self.is_alpha_updated[a] = 1
            
        for a in set_to_alpha[deployed_set]:
            if not self.is_alpha_updated[a]:
                self.update_counters(a, int(observed_reward))
                self.is_alpha_updated[a] = 1
        
        self.arm_t[self.current_time_step] = alpha_to_deploy        
        self.current_time_step+=1
        
    def collect_rewards_for_each_alpha(self, alphas_to_search):
        for alpha in alphas_to_search:
            if not np.isnan(alpha):
                if not self.is_alpha_updated[alpha]:
                    self.update_rewards(alpha)
            if self.current_time_step == self.T:
                return

    def deactivate_rule(self, active_alphas_mask):
        alphas_ucbs = self.ucbs(self.conf_predictors.alpha_values)
        max_lcb = self.lcbs(self.conf_predictors.alpha_values).max()

        for i,_ in enumerate(active_alphas_mask):
            if not (alphas_ucbs[i] >= max_lcb):
                active_alphas_mask[i] = 1

        return active_alphas_mask

    def run_algorithm(self):
        active_alphas_mask = np.zeros_like(self.conf_predictors.alpha_values).astype(bool)
        masked_alphas = ma.masked_array(self.conf_predictors.alpha_values, mask=active_alphas_mask)
        while self.current_time_step < self.T and active_alphas_mask.sum() < (len(active_alphas_mask)-1):
            masked_alphas = ma.masked_array(self.conf_predictors.alpha_values, mask=active_alphas_mask)
            masked_alphas = ma.filled(masked_alphas, np.nan)
            self.collect_rewards_for_each_alpha(masked_alphas)
            active_alphas_mask = self.deactivate_rule(active_alphas_mask)
            self.is_alpha_updated = defaultdict(lambda:0)
        
        print(f"{len(active_alphas_mask) - active_alphas_mask.sum()} active arms")

        # If many alpha values remain active return the one with the highest lcb
        best_alpha_idx = self.lcbs(self.conf_predictors.alpha_values).argmax()
        best_alpha_found = self.conf_predictors.alpha_values[best_alpha_idx]
        
        masked_alphas = ma.masked_array(self.conf_predictors.alpha_values, mask=active_alphas_mask)
        print(masked_alphas)
        avs = []
        n_rs = []
        for a in masked_alphas.compressed():
            avs.append(self.average_reward(a))
            n_rs.append(self.n_rewards[a])
        print(avs)
        print(n_rs)
        
        # Save arms per timestep 
        path = f"{config.ROOT_DIR}/{config.results_path}/regret/{config.algorithm_key}/seed_{config.run_no_seed}_cal_{config.run_no_cal}.npy"
        np.save(path, self.arm_t)
        return best_alpha_found,best_alpha_idx

