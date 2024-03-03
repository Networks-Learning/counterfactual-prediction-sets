import numpy as np
import config
from collections import defaultdict

"""Implements all conformal predictors given a fixed calibration set"""

class ConformalPrediction:
    def __init__(self, x_cal, y_cal, model) -> None:
        self.x_cal = x_cal
        self.y_cal = y_cal
        self.model = model
        self.calibration_set_size = len(self.x_cal)
        
        # Alpha values and quantile probabilities
        self.alpha_values = 1 - np.arange(1, self.calibration_set_size + 1) / (self.calibration_set_size + 1)
        self.quantile_probabilities = np.ceil((1 - self.alpha_values)*(self.calibration_set_size + 1)) / self.calibration_set_size
        
        # Conformal scores on true labels
        model_output = self.model.pred_prob(self.x_cal)
        true_label_one_hot = np.eye(config.N_LABELS)[self.y_cal]

        true_label_logits = model_output * true_label_one_hot
        true_label_conf_scores = np.sort(1 - true_label_logits.sum(axis=1))

        # Quantiles for each alpha value
        self.quantiles = np.quantile(true_label_conf_scores, self.quantile_probabilities)
        self.quantiles_dict = {alpha : self.quantiles[i] for i, alpha in enumerate(self.alpha_values)}

        # Add human alone baseline
        self.quantiles_dict[0] = np.inf
        self.alpha_values = np.concatenate((self.alpha_values, [0]))

    def set_sizes_alphas(self, x):
        "Returns a dictionary of set sizes to alpha values and vice versa for a sample"
        # pred_prob_sorted returns the pred_prob sorted in descending order 
        conf_scores_sorted = 1 - self.model.pred_prob_sorted(x)
        set_sizes_to_alphas = defaultdict(lambda:[])
        alphas_to_set_sizes = {}
        # Quantile index
        j = 0 
        for i in range(1, config.N_LABELS):
            while self.quantiles[j] < conf_scores_sorted[i-1]:
                set_sizes_to_alphas[i-1].append(self.alpha_values[j])
                alphas_to_set_sizes[self.alpha_values[j]] = i - 1
                j+=1
                if j == len(self.quantiles):
                    break
            if j == len(self.quantiles):
                break 
        while j < len(self.quantiles):
            set_sizes_to_alphas[config.N_LABELS].append(self.alpha_values[j])
            alphas_to_set_sizes[self.alpha_values[j]] = config.N_LABELS
            j+=1
        set_sizes_to_alphas[config.N_LABELS].append(0)
        alphas_to_set_sizes[0] = config.N_LABELS
        
        return set_sizes_to_alphas, alphas_to_set_sizes   