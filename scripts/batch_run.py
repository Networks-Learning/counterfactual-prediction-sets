import subprocess
from copy import deepcopy
from tqdm import tqdm
import os
from utils import create_path
import config

"""Executes all bandit algorithms for config.n_runs number of runs"""

args_base = ["python", "-m", "scripts.run_bandit", "--cal_run", "1","--seed_run"]

for seed_run in tqdm(range(config.n_runs)):
    for alg in [ "SE", "SE_no_mon", "SE_ours", "UCB", "UCB_no_mon", "UCB_ours"]:
        args = deepcopy(args_base)
        args.append(f"{seed_run}")
        args.append("--alg")
        args.append(alg)
        if config.violations > 0:
            args.append("--pv")
            args.append(f"{config.violations}")
        output_dir = f"{config.output_path}/{alg}"
        create_path(output_dir)
        output_file = f"{output_dir}/{alg}_seed_run{seed_run}_cal_run1.out"
        subprocess.run(["touch", output_file])
        with open(output_file, 'wt') as f:
            subprocess.run(args, stdout=f)