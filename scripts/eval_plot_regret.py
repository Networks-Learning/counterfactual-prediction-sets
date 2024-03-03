import subprocess
import os
import config

"""Script to reproduce the regret plot"""

if not os.path.exists(f"{config.ROOT_DIR}/{config.results_path}/strict/opt_avg_acc_se_alphas_1.json"):
    # Compute the empirical expert success probability for each alpha value under the strict implementation given a calibration set under optimization mode
    args_base = ["python", "-m", "scripts.run_bandit", "--cal_run", "1","--alg", "test", "--mode","optimize"]
    subprocess.run(args=args_base)
    # The above runs
    # python3 -m scripts.run_bandit --cal_run 1 --alg test --mode optimize

# Plot the empirical expected regret for all bandit algorithms that ran
args = ["python", "-m", "plotters.regret", "--cal_run", "1", "--mode", "optimize"]
subprocess.run(args=args)
# The above runs
# python3 -m plotters.regret --cal_run 1 --mode optimize
