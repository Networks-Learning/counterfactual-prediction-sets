import os
import numpy as np
import argparse

"""Experiments configuration"""

ROOT_DIR = os.path.dirname(__file__)
MODEL = 'vgg19'
NOISE_LEVEL = 110
N_LABELS = 16

parser = argparse.ArgumentParser()
parser.add_argument("--seed_run", type=int, help='Select random seed', default=0) 
parser.add_argument("--cal_run", type=int, help='Select random state for data split', default=1) 
parser.add_argument("--alg", choices=["SE_ours", "SE_no_mon", "SE", "UCB_ours", "UCB_no_mon", "UCB", "test"], \
                    default='test',\
                    help='Choose the algorithm to run or compute the empirical expert success probability per alpha value under the strict implementation (option "test").') 
parser.add_argument("--mode", choices=['optimize', 'deploy'], default='deploy',\
                    help="Mode of operation to be used only during evaluation and plots. Use 'optimize' for evaluation and plots about regret and 'deploy' for evaluation and plots about deployment under the strict and the lenient implementation.")
parser.add_argument("--pv", type=float, help='Set the p_v value, that is the fraction of the data with violations of interventional monotonicity.', default=0)
args,unknown = parser.parse_known_args()

# Differentiates the seed for every run 
run_no_seed = args.seed_run
# Differentiates the calibration split
run_no_cal = args.cal_run

# Initialize random generators
seed = 7654174832901 + run_no_seed
numpy_rng = np.random.default_rng(seed=seed)
numpy_rng_option_shuffler = np.random.default_rng(seed=seed)

# Fraction of the dataset to use as calibration set
calibration_split = 0.1

# Algorithm to run
algorithm_key = args.alg

# Fraction of data with induced violations of monotonicity
# MUST be 0 for experiments with the original human subject study dataset 
violations = args.pv

# Path for dataset with randomly permuted rewards to violate interventional monotonicity
violations_path = f"robustness/datasets/permutation_violations"

# Dataset with observed rewards based on the study data 
rewards_file = f"study_data/rewards" 

# Select processed datasets for sensitivity analysis
if violations > 0: 
    rewards_file = f"{violations_path}/{violations}_violations"

# Set up paths for results, output logs and plots
# Path to store results
results_path = f"results" 
# Path to store output logs
output_path = f"output"
# Path to store plots
plot_path = f"plots"

# Diversify path for sensitivity analysis if necessary
fix_path_fn = lambda path_root_name,base_path,setting_key : f"{path_root_name}/{base_path}" if setting_key > 0 else path_root_name

violations_base_path = f"violations/permutation_violations/{violations}_violations"

results_path = fix_path_fn(results_path,violations_base_path,violations) 
output_path = fix_path_fn(output_path,violations_base_path,violations)
plot_path = fix_path_fn(plot_path,violations_base_path,violations)
    
# Total number of realizations of the bandit algorithms
n_runs = 30

# Flag for operation mode. Set to True for evaluation and plots for the regret analysis. 
regret = True if args.mode == 'optimize' else False
avg_acc_se_alphas = "opt_avg_acc_se_alphas_" if args.mode == 'optimize' else "deploy_avg_acc_se_alphas_"