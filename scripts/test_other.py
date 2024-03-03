from utils import split_dataset, create_path
import config
from models.model import Model
from conformal_prediction import ConformalPrediction
from algorithms.test import Test
import pandas as pd
import numpy as np
import json

"""Computes the empirical expert success probability or each alpha value under the lenient implementation"""

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
reward_path = f"{config.ROOT_DIR}/{config.rewards_file}_lenient.csv"
rewards = pd.read_csv(reward_path, dtype={'reward': np.bool_}).set_index('image_name')

assert not all(rewards.reward)

# Initialize algorithm
bandits_alg = Test(X_train, conformal_predictors, rewards)

# Compute Avg reward and se for each alpha
average_acc_per_alpha, se_acc_per_alpha = bandits_alg.run_algorithm()

alpha_values = bandits_alg.conf_predictors.alpha_values
results_dict = { 
    alpha: { 
        'avg': average_acc_per_alpha[i],
        'se': se_acc_per_alpha[i],
    } 
    for i,alpha in enumerate(alpha_values) 
}
results_path = f"{config.ROOT_DIR}/{config.results_path}/lenient/{config.avg_acc_se_alphas}{config.run_no_cal}.json"
create_path(f"{config.ROOT_DIR}/{config.results_path}/lenient/")

with open(results_path, 'wt') as f:
        json.dump(results_dict, f)
