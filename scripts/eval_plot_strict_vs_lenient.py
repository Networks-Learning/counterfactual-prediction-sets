import subprocess
import config
import os

"""Script to reproduce plots comparing the strict and the lenient implementation"""

# Compute the empirical expert success probability for each alpha value under the strict implementation given a calibration set under deployment mode
if not os.path.exists(f"{config.ROOT_DIR}/{config.results_path}/strict/deploy_avg_acc_se_alphas_1.json"):
    args_base = ["python", "-m", "scripts.run_bandit", "--cal_run", "1","--alg", "test", "--mode","deploy"]
    subprocess.run(args=args_base)

# Compute the empirical expert success probability for each alpha value under the lenient implementation given a calibration set under deployment mode
if not os.path.exists(f"{config.ROOT_DIR}/{config.results_path}/lenient/deploy_avg_acc_se_alphas_1.json"):
    args_base = ["python", "-m", "scripts.test_other", "--cal_run", "1", "--mode","deploy"]
    subprocess.run(args=args_base)

# Compute the number of predictions in which the humans misplaced their trust and the number of predictions in which they predicted correctly when the true label was not in the prediction set under the lenient implementation
if not os.path.exists(f"{config.ROOT_DIR}/{config.results_path}/lenient/n_misplaced_trust_1.json") and config.violations == 0:
    args_base = ["python", "-m", "scripts.misplaced_trust_loss", "--cal_run", "1", "--mode","deploy"]
    subprocess.run(args=args_base)

# Produce the plots for the above and highlight the alpha values found by competitive baselines highlighted in the paper
args = ["python", "-m", "plotters.lenient", "--cal_run", "1", "--seed_run", "27", "--mode", "deploy"]
subprocess.run(args=args)