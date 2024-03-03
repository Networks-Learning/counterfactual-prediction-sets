from utils import split_dataset, create_path
import config
from models.model import Model
from conformal_prediction import ConformalPrediction
from algorithms.successive_elimination import SuccessiveEliminationOurs, SuccessiveEliminationNoMonotonicity, SuccessiveElimination
from algorithms.ucb import UCBNoMonotonicity, UCBOurs, UCB
from algorithms.test import Test
import pandas as pd
import numpy as np
import json
import os

"""Executes one realization of one bandit algorithm"""

# Algorithms
algorithms = {
    "SE_ours": SuccessiveEliminationOurs,
    "SE_no_mon": SuccessiveEliminationNoMonotonicity,
    "SE": SuccessiveElimination,
    "UCB_ours": UCBOurs,
    "UCB_no_mon": UCBNoMonotonicity,
    "UCB": UCB,
    "test": Test
}

# Split the dataset
X_train, X_cal, y_train, y_cal = split_dataset(config.run_no_cal)

# Shuffle train set
config.numpy_rng.shuffle(X_train)

# Total number of timesteps
config.total_timesteps = len(X_train)

# Create model
model = Model()

# Initialize predictors
conformal_predictors = ConformalPrediction(X_cal, y_cal, model)

# Get observed rewards for training
reward_path = f"{config.ROOT_DIR}/{config.rewards_file}_strict.csv"
rewards = pd.read_csv(reward_path, dtype={'reward': np.bool_}).set_index('image_name')

assert not all(rewards.reward)

# Initialize algorithm
bandits_alg = algorithms[config.algorithm_key](X_train, conformal_predictors, rewards)

# Check if in testing mode
if config.algorithm_key != 'test':
    
    # Create path to store the selected arm per time step
    path = f"{config.ROOT_DIR}/{config.results_path}/regret/{config.algorithm_key}"
    create_path(path)
    # Create path to store running logs
    path = f"{config.ROOT_DIR}/{config.output_path}/{config.algorithm_key}"
    create_path(path)
    # Run the algorithm 
    best_alpha_found, best_alpha_idx = bandits_alg.run_algorithm()

    # Save alpha_value and alpha_index
    results_path = f"{config.ROOT_DIR}/{config.results_path}/alg_alphas.csv"
    results_df = pd.DataFrame(data=[(config.algorithm_key, best_alpha_idx, best_alpha_found, config.run_no_cal, len(X_cal))], columns=["Algorithm", "Alpha index", "Alpha value", "Cal_Run", "Calibration set size"])
    if os.path.isfile(results_path):
        results_df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_path, header=True, index=False)

else:
    # Compute Avg reward and standard error for each alpha
    average_acc_per_alpha, se_acc_per_alpha = bandits_alg.run_algorithm(regret=config.regret)

    alpha_values = bandits_alg.conf_predictors.alpha_values
    results_dict = { 
        alpha: { 
            'avg': average_acc_per_alpha[i],
            'se': se_acc_per_alpha[i]
        } 
        for i,alpha in enumerate(alpha_values) 
    }
    results_path = f"{config.ROOT_DIR}/{config.results_path}/strict/{config.avg_acc_se_alphas}{config.run_no_cal}.json"
    
    # Create results path
    create_path(f"{config.ROOT_DIR}/{config.results_path}/strict")

    with open(results_path, 'wt') as f:
        json.dump(results_dict, f)
