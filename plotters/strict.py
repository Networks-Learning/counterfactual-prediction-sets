import config
import json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plotters.lenient import marker_points, avg_acc_active_arms
from utils import create_path

my_cmap =  sns.color_palette('colorblind')
legend_font_size = 25
marker_size_main = 15
marker_size_hat = 20
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts,geometry}'
mpl.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams.update({
    'font.family':'serif',
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "font.size": 38,
    "figure.figsize":(15,9)
})
marker_size_hat = 200

def alpha_vs_acc(show_full=False, save=True):
    """Empirical success probability (accuracy) for each alpha 
    value under the strict implementation of our system"""
    path = f"{config.ROOT_DIR}/{config.results_path}/strict/{config.avg_acc_se_alphas}{config.run_no_cal}.json"
    with open(path, 'rt') as f:
        strict_alphas_avg_acc_se = json.load(f)
    
    df_strict = pd.DataFrame(strict_alphas_avg_acc_se).T
    df_strict.rename(columns={'avg':'avg_strict', 'se':'se_strict'}, inplace=True)
    df_strict.index.rename('alpha', inplace=True)
    print(df_strict)    

    df_strict.sort_index(inplace=True)
    
    alphas_to_plot = df_strict[df_strict.index < '0.5']
    if show_full:
        alphas_to_plot = df_strict
    # Highlight only optimal alpha and
    # alpha returned by counterfactual UCB1 
    marker_alpha = marker_points(alphas_to_plot)[1:-1]

    alphas_to_plot.index = alphas_to_plot.index.astype(float)
    ci_radious = alphas_to_plot[['se_strict']]*1.96
    
    ax = alphas_to_plot.plot(y=['avg_strict'], style=['-o'], color=[my_cmap[0]], markersize=marker_size_main, zorder=0)
   
    # Set marker colors for baselines
    for i,j in zip(alphas_to_plot.index, alphas_to_plot['avg_strict'].values): 
        color = my_cmap[0] if i not in marker_alpha else 'darkblue'
        markersize = marker_size_main if i not in marker_alpha else marker_size_hat
        marker = 'o'

        ax.scatter(i,j, color=color, marker=marker, s=markersize, zorder=1)    

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r"Empirical Success Probability")
    ax.spines[['right', 'top']].set_visible(False)
    # Plot average accuracy of active arms 
    # after running counterfactual SE as an horizontal line
    avg_acc_arms, se_acc_arms = avg_acc_active_arms(strict_alphas_avg_acc_se, run_no_cal=config.run_no_cal, run_no_seed=config.run_no_seed)
    xmin = alphas_to_plot.index.values.min()
    xmax = 10
    ax.axhline(avg_acc_arms, xmin=xmin, xmax=xmax, linestyle='--', color=my_cmap[2])
    print(1.96*se_acc_arms)
    ax.legend().set_visible(False)

    # Fix y axis borders for datasets with violations
    if config.violations > 0:
        ylim_low = 0.75
        ylim_high = 0.88
        ax.set_ylim(bottom=ylim_low, top=ylim_high)
    
    # Add 95% CI in line strict
    setting = 'strict'
    avg_k = f"avg_{setting}"
    se_k = f"se_{setting}"
    y1 = alphas_to_plot[avg_k].values-ci_radious[se_k].values
    y2 = alphas_to_plot[avg_k].values+ci_radious[se_k].values  
    ax.fill_between(alphas_to_plot.index, y1=y1, y2=y2, alpha=.2)
    if save:
        base_path = f"{config.ROOT_DIR}/{config.plot_path}/strict/"  
        create_path(base_path)
        path = f"{base_path}/acc_vs_alpha_cal_run_{config.run_no_cal}_seed_run_{config.run_no_seed}{'_full' if show_full else ''}.pdf"
        plt.savefig(path, bbox_inches='tight')
    # plt.show()
    
if __name__=='__main__':
    for show_f in [True, False]:
        alpha_vs_acc(show_full=show_f, save=True)
        