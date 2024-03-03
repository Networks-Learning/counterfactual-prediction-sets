import subprocess
import config
import os

"""Script to reproduce performance plots for the strict implementation of our system"""

# Compute the empirical expert success probability for each alpha value under the strict implementation given a calibration set under deployment mode
if not os.path.exists(f"{config.ROOT_DIR}/{config.results_path}/strict/deploy_avg_acc_se_alphas_1.json"):
    args_base = ["python", "-m", "scripts.run_bandit", "--cal_run", "1","--alg", "test", "--mode","deploy"]
    if config.violations > 0:
        args_base.append("--pv")
        args_base.append(f"{config.violations}")
    subprocess.run(args=args_base)


# Produce the plots for the above and highlight the alpha values found 
# by counterfactual UCB1, counterfactual SE, and the optimal alpha
args = ["python", "-m", "plotters.strict", "--cal_run", "1", "--seed_run", "24", "--mode", "deploy"]
if config.violations > 0:
    args.append("--pv")
    args.append(f"{config.violations}")
subprocess.run(args=args)