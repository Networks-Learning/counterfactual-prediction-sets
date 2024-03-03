from algorithms.algorithm import Algorithm
from algorithms.utils import *
from models.preprocess_predictions import get_idx_min_valid_non_singleton_set as pivot_sets
from collections import defaultdict

class MisplacedTrustLoss(Algorithm):
    def __init__(self, data_samples, conformal_predictors, rewards) -> None:
        super().__init__(data_samples, conformal_predictors, rewards)
        # Smallest sets containing the true label
        self.pivot_sets = pivot_sets()
        # Counter: prediction from the prediction set when the set is invalid
        self.mispredictions_from_set = defaultdict(lambda:0)
        # Counter: prediction from outside the prediction set when the set is valid
        self.mispredictions_outside_set = defaultdict(lambda:0)
        # Counter: total number of predictions
        self.n_predictions = defaultdict(lambda:0)
        # Counter: number of correct predictions when the sets are invalid
        self.n_correct_invalid = defaultdict(lambda:0)
        
    def run_algorithm(self):
        """Counts for each alpha value the number of predictions in 
            which the prediction sets are invalid and the experts succeed,
            the number of predictions in which the prediction sets are valid
            and the experts predict from outside the set,
            and the number of predictions in which the prediction sets,
            are invalid and the experts predict from the sets.
            """
        while self.current_time_step<self.T:
            data_sample = self.data[self.current_time_step]
            pivot_set = self.pivot_sets[data_sample] + 1
            set_to_alpha, _ = self.conf_predictors.set_sizes_alphas(data_sample)
            rew_slice = self.rewards.loc[data_sample][['set', 'reward', 'is_other']]
            n_human_alone = len(rew_slice[rew_slice['set'] == 16])
            for s in set_to_alpha:
                set_q = rew_slice['set'] == s
                from_set_q = rew_slice['is_other'] == False
                outside_set_q = rew_slice['is_other'] == True
                rew_true_q = rew_slice['reward'] == True
                mispred_from_set = rew_slice[(set_q)&(from_set_q)]
                mispred_outside_set = rew_slice[(set_q)&(outside_set_q)]
                success_outside_set = rew_slice[(set_q)&(outside_set_q)&(rew_true_q)]
                n_pred = len(rew_slice[(set_q)])
                for a in set_to_alpha[s]:
                    if not s:
                        self.n_predictions[a]+=n_human_alone
                    else:
                        if s < pivot_set:
                            self.mispredictions_from_set[a]+=len(mispred_from_set) 
                            self.n_correct_invalid[a]+=len(success_outside_set)

                        else:
                            self.mispredictions_outside_set[a]+=len(mispred_outside_set)
                        self.n_predictions[a]+=n_pred
                        
            self.current_time_step+=1
          
        return self.mispredictions_from_set, self.mispredictions_outside_set, self.n_predictions, self.n_correct_invalid
