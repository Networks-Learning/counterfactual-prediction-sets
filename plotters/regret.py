import config
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import seaborn as sns
from utils import create_path

legend_font_size = 25
marker_size_main = 15

my_cmap =  sns.color_palette('colorblind')
sns.set_palette(sns.color_palette('colorblind'))
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts,geometry}'
mpl.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams.update({
    'font.family':'serif',
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "font.size": 38,
    "figure.figsize":(15,9)
})

def read_data(n_runs=config.n_runs):
    alg_run_mean_rt = defaultdict(lambda:{})
    alg_run_se_rt = defaultdict(lambda:{})
    results_path_root = f"{config.ROOT_DIR}/{config.results_path}/regret/"

    path = f"{config.ROOT_DIR}/{config.results_path}/strict/{config.avg_acc_se_alphas}{config.run_no_cal}.json"
    with open(path, 'rt') as f:
        alphas_avg_acc_se = json.load(f)
    alphas_avg_acc_se_df = pd.DataFrame(data=alphas_avg_acc_se).T
    
    opt_mean_r = alphas_avg_acc_se_df['avg'].max()

    for root, dirs, files in os.walk(results_path_root):
        for dir in dirs:
            for _, _, files in os.walk(f"{root}/{dir}/"):
                for file in files:
                    if file.split('.')[0].split('_')[-1] != str(config.run_no_cal):
                        continue
                    arm_per_t = np.load(f"{root}/{dir}/{file}")
                    alg = dir
                    run = int(file.split('_')[1])
                    if not (run < n_runs):
                        continue
                    alg_run_mean_rt[alg][run] = np.array([alphas_avg_acc_se_df.loc[str(arm), 'avg'] for arm in arm_per_t])
                    alg_run_mean_rt['opt'][run] = opt_mean_r*np.ones_like(arm_per_t)
                    alg_run_se_rt[alg][run] = np.array([alphas_avg_acc_se_df.loc[str(arm), 'se'] for arm in arm_per_t])
    
    return alg_run_mean_rt, alg_run_se_rt

def compute_regret(n_runs=config.n_runs):
    alg_run_mean_rt, alg_run_se_rt = read_data(n_runs)
    # Get expected reward for each arm 
    alg_run_mean_rt_df = pd.DataFrame(data=alg_run_mean_rt).sort_index()
    alg_run_mean_rt_df.index.name = 'Run'
    
    alg_cum_run_mean_rw = alg_run_mean_rt_df.applymap(np.cumsum)
    # Mean reward per timestep across runs
    stack_series_mean = lambda series: np.stack(series.tolist()).mean(axis=0)
    alg_cum_mean_rw = alg_cum_run_mean_rw.agg(stack_series_mean, axis=0)
    
    # Standard error of mean reward across runs
    stack_series_errors = lambda series: np.stack(series.tolist()).std(axis=0)/np.sqrt(series.tolist()[1].shape)
    alg_cum_se_rw = alg_cum_run_mean_rw.agg(stack_series_errors, axis=0).drop('opt', axis=1)
    assert all(alg_cum_se_rw < .2)
    for alg in alg_cum_mean_rw.columns:
        if alg!='opt':
            alg_cum_mean_rw[alg] = alg_cum_mean_rw['opt'] - alg_cum_mean_rw[alg]
        
    return alg_cum_mean_rw.drop('opt', axis=1),alg_cum_se_rw

def plot_regret(n_runs=config.n_runs, save=True):
    regret_old_col, err = compute_regret(n_runs)
    print(err)
    # Fix order of columns
    regret_old_col = regret_old_col[['SE', 'SE_no_mon', 'UCB', 'UCB_no_mon', 'SE_ours', 'UCB_ours']]
    regret = regret_old_col.rename({
        'SE':'SE',
        'SE_no_mon': 'AF Counterfactual SE',
        'UCB': r"\texttt{UCB1}",
        'UCB_no_mon': r"AF Counterfactual \texttt{UCB1}",
        'SE_ours': 'Counterfactual SE',
        'UCB_ours': r'Counterfactual \texttt{UCB1}'
    },axis=1)
    styles = ['--','-.', '-', ':', '.--','*--']
    for i, col in enumerate(regret.columns):
        if not i:
            ax = regret[[col]].plot(logy=True,label=col, style=styles[i], color=my_cmap[i])
        else:
            regret[[col]].plot(label=col, style=styles[i], color=my_cmap[i], ax=ax, markersize=marker_size_main, markevery=50)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r'Empirical Average Regret')
    ax.legend(fontsize=legend_font_size, ncols=2)
    ax.spines[['right', 'top']].set_visible(False)

    ci_radious = err
    print(ci_radious)

    for i, col in enumerate(regret_old_col.columns):
        y1 = regret_old_col[col].values-ci_radious[col].values
        y2 = regret_old_col[col].values+ci_radious[col].values
        ax.fill_between(regret_old_col.index, y1=y1, y2=y2, alpha=.2)

    if save:
        regret_path = f"{config.ROOT_DIR}/{config.plot_path}/regret"
        create_path(regret_path)
        path = f"{regret_path}/n_runs_{n_runs}_cal_run_{config.run_no_cal}.pdf"
        plt.savefig(path, bbox_inches='tight')
    # plt.show()
    return regret

if __name__=='__main__':
    plot_regret(save=True)

