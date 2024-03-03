from utils import split_dataset,create_path
import config
from models.model import Model
from conformal_prediction import ConformalPrediction
from algorithms.misplaced_trust_loss import MisplacedTrustLoss
import pandas as pd
import numpy as np
import json

"""Computes for each alpha value, the number of predictions 
   in which the experts predict the correct label from outside
   the prediction sets when the prediction sets do not contain 
   the ground truth label, the number of predictions in which 
   the experts predict from outside (from) the prediction sets, 
   when the prediction sets do (not) contain the ground truth 
   label, as well as the total number of predictions. 
   """

# Split the dataset
X_train, X_cal, y_train, y_cal = split_dataset(config.run_no_cal)

# Shuffle train set
config.numpy_rng.shuffle(X_train)

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
bandits_alg = MisplacedTrustLoss(X_train, conformal_predictors, rewards)

# Compute number of mispredictions due to misplaced trust
mispredictions_from_set, mispredictions_outside_set, n_predictions, n_correct_invalid_set = bandits_alg.run_algorithm()
alpha_values = bandits_alg.conf_predictors.alpha_values
results_dict = { 
    alpha: { 
        'n_mispredictions_from_set': mispredictions_from_set[alpha],
        'n_mispredictions_outside_set': mispredictions_outside_set[alpha],
        'n_correct_invalid_set': n_correct_invalid_set[alpha],
        'n_predictions': n_predictions[alpha]
    } 
    for alpha in alpha_values
}
results_path = f"{config.ROOT_DIR}/{config.results_path}/lenient/n_misplaced_trust_{config.run_no_cal}.json"
create_path(f"{config.ROOT_DIR}/{config.results_path}/lenient/")
with open(results_path, 'wt') as f:
        json.dump(results_dict, f)
        