import config
import json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

def marker_points(strict_alphas_avg_acc_se_df):
    """Returns the alpha values to highlight in the plot""" 
    # Alpha value of the offline baseline
    # that uses a stylized expert model
    alpha_offline = 0.016528925619834656

    # Alpha used in Babbar et al.
    alpha_babbar = 0.1

    # Get the optimal alpha
    optimal_alpha = strict_alphas_avg_acc_se_df['avg_strict'].idxmax()

    # Our alpha
    path = f"{config.ROOT_DIR}/{config.results_path}/alg_alphas.csv"
    alg_alphas_df = pd.read_csv(path)
    run_q = alg_alphas_df['Cal_Run'] == config.run_no_cal
    alg_q = alg_alphas_df['Algorithm'] == 'UCB_ours'

    alpha_ours = alg_alphas_df[(run_q)&(alg_q)]['Alpha value'].values[config.run_no_seed]
    if alpha_ours not in strict_alphas_avg_acc_se_df.index:
        alpha_ours_idx = np.searchsorted(strict_alphas_avg_acc_se_df.index.astype(float), float(alpha_ours), side='left')
        alpha_ours = strict_alphas_avg_acc_se_df.index[alpha_ours_idx]

    if alpha_babbar not in strict_alphas_avg_acc_se_df.index:
        alpha_babbar_idx = np.searchsorted(strict_alphas_avg_acc_se_df.index.astype(float), float(alpha_babbar), side='left')
        alpha_babbar = strict_alphas_avg_acc_se_df.index[alpha_babbar_idx]

    x = [float(alpha_offline), float(alpha_ours), float(optimal_alpha), float(alpha_babbar)]
    return x

def avg_acc_active_arms(strict_alphas_avg_acc_se, run_no_seed=0, run_no_cal=0):
    """Returns the average accuracy of the arms that remained active at the last round
    after running counterfactual SE"""
    path = f"{config.ROOT_DIR}/{config.output_path}/SE_ours/active_arms_run{run_no_seed}_calrun{run_no_cal}.json"
    with open(path, 'rt') as f:
        active_arms = json.load(f)
    print(active_arms)
    print(len(active_arms))
    accuracies = np.array([strict_alphas_avg_acc_se[str(a)]['avg'] for a in active_arms])
    print(accuracies)
    print(accuracies.mean())
    return accuracies.mean(), accuracies.std()/np.sqrt(len(accuracies))

def alpha_vs_acc(show_full=False, save=True, show_markers=True, show_cf_se=True, no_baselines=False):
    """Plot alpha vs accuracy for the strict and the lenient implementation of our systems 
    adn highlight baselines"""
    path = f"{config.ROOT_DIR}/{config.results_path}/strict/{config.avg_acc_se_alphas}{config.run_no_cal}.json"
    with open(path, 'rt') as f:
        strict_alphas_avg_acc_se = json.load(f)
    
    df_strict = pd.DataFrame(strict_alphas_avg_acc_se).T
    df_strict.rename(columns={'avg':'avg_strict', 'se':'se_strict'}, inplace=True)
    df_strict.index.rename('alpha', inplace=True)
    print(df_strict)    

    path = f"{config.ROOT_DIR}/{config.results_path}/lenient/{config.avg_acc_se_alphas}{config.run_no_cal}.json"
    with open(path, 'rt') as f:
        lenient_alphas_avg_acc_se = json.load(f)
    
    df_lenient = pd.DataFrame(lenient_alphas_avg_acc_se).T
    df_lenient.rename(columns={'avg':'avg_lenient', 'se':'se_lenient'}, inplace=True)
    df_lenient.index.rename('alpha', inplace=True)

    all_df = df_strict.merge(df_lenient, on='alpha', how='outer').sort_index()
    
    alphas_to_plot = all_df[all_df.index < '0.5']
    if show_full:
        alphas_to_plot = all_df
    
    alphas_to_plot.index = alphas_to_plot.index.astype(float)
    ci_radious = alphas_to_plot[['se_strict', 'se_lenient']]*1.96
    
    ax = alphas_to_plot.plot(y=['avg_strict','avg_lenient'], style=['-o', '-x'], color=[my_cmap[0], my_cmap[1]], markersize=marker_size_main, zorder=0)
   
    if show_markers:
        marker_alpha = marker_points(alphas_to_plot)
        if no_baselines:
            marker_alpha = marker_alpha[1:-1]
        # Set marker colors for baselines
        for i,j in zip(alphas_to_plot.index, alphas_to_plot['avg_strict'].values): 
            color = my_cmap[0] if i not in marker_alpha else 'darkblue'
            markersize = marker_size_main if i not in marker_alpha else marker_size_hat
            marker = 'o'
            # Mark lenient
            if i == marker_alpha[-1] and not no_baselines:
                color = 'saddlebrown'
                j = alphas_to_plot.loc[i,'avg_lenient']
                marker='x'

            ax.scatter(i,j, color=color, marker=marker, s=markersize, zorder=1)    

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r"Empirical Success Probability")
    ax.spines[['right', 'top']].set_visible(False)
    
    if show_cf_se:
        # Plot avg acc active as horizontal line
        avg_acc_arms, se_acc_arms = avg_acc_active_arms(strict_alphas_avg_acc_se, run_no_cal=config.run_no_cal, run_no_seed=config.run_no_seed)
        xmin = alphas_to_plot.index.values.min()
        xmax = 10
        ax.axhline(avg_acc_arms, xmin=xmin, xmax=xmax, linestyle='--', color=my_cmap[2])
        print(1.96*se_acc_arms)
    ax.legend([ r'Strict', r'Lenient'])
        
    # Add 95% CI in lines strict and lenient
    for setting in ['strict', 'lenient']:
        avg_k = f"avg_{setting}"
        se_k = f"se_{setting}"
        y1 = alphas_to_plot[avg_k].values-ci_radious[se_k].values
        y2 = alphas_to_plot[avg_k].values+ci_radious[se_k].values  
        ax.fill_between(alphas_to_plot.index, y1=y1, y2=y2, alpha=.2)
    if save:
        path = f"{config.ROOT_DIR}/{config.plot_path}/lenient/acc_vs_alpha_cal_run_{config.run_no_cal}_seed_run_{config.run_no_seed}{'_full' if show_full else ''}{'_nobaselines' if no_baselines else ''}.pdf"
        create_path(f"{config.ROOT_DIR}/{config.plot_path}/lenient/")
        plt.savefig(path, bbox_inches='tight')
    # plt.show()
    
def invalid_sets_vs_misplaced_trust(save=False, disadvantage=False):
    """Plot for each alpha the number of times the experts 
     misplaced their trust and the number of times they predicted
      correctly when the prediction set did not include the 
      correct label under the lenient implementation"""
    path = f"{config.ROOT_DIR}/{config.results_path}/lenient/n_misplaced_trust_{config.run_no_cal}.json"
    with open(path, 'rt') as f:
        n_misplaced_trust = json.load(f)
    
    df_n_misplaced_trust = pd.DataFrame(n_misplaced_trust).T
    if disadvantage:
        df_n_mispred = df_n_misplaced_trust['n_mispredictions_outside_set']
    else:
        df_n_mispred = df_n_misplaced_trust['n_mispredictions_outside_set'] + df_n_misplaced_trust['n_mispredictions_from_set']
    df_misplaced_trust = df_n_mispred 
    df_misplaced_trust.index.rename('alpha', inplace=True)
    df_misplaced_trust = df_misplaced_trust.to_frame()
    
    print(df_misplaced_trust)  

    df_correct_inv_sets = df_n_misplaced_trust['n_correct_invalid_set'] 
    df_correct_inv_sets.index.rename('alpha', inplace=True)
    df_correct_inv_sets = df_correct_inv_sets.to_frame()

    all_df = df_correct_inv_sets.merge(df_misplaced_trust, on='alpha', how='outer').sort_index()
    
    alphas_to_plot = all_df
    print(all_df)
    alphas_to_plot.index = alphas_to_plot.index.astype(float)
    ax = alphas_to_plot.plot(style=['-o', '-x'], color=[my_cmap[0], my_cmap[1]], markersize=marker_size_main, zorder=0)
    ax.set_xlabel(r'$\alpha$')
    ax.spines[['right', 'top']].set_visible(False)
    sum_of_correct_preds_inv_set = r'$\sum_{x,y}\mathbb{I}\{\hat{Y}_{\mathcal{C}_{\alpha}} = y \wedge y \notin \mathcal{C}_{\alpha}(x)\}$'
    if disadvantage:
        sum_of_incorrect_misplaced_trust = r'$\sum_{x,y}\mathbb{I}\{\hat{Y}_{\mathcal{C}_{\alpha}} \notin \mathcal{C}_{\alpha}(x) \wedge y \in \mathcal{C}_{\alpha}(x)\}$'
    else:    
        sum_of_incorrect_misplaced_trust = r'$\sum_{x,y}\mathbb{I}\{\hat{Y}_{\mathcal{C}_{\alpha}} \notin \mathcal{C}_{\alpha}(x) \wedge y \in \mathcal{C}_{\alpha}(x)\} + \mathbb{I}\{\hat{Y}_{\mathcal{C}_{\alpha}} \in \mathcal{C}_{\alpha}(x) \wedge y \notin \mathcal{C}_{\alpha}(x)\}$'
    ax.legend([sum_of_correct_preds_inv_set, sum_of_incorrect_misplaced_trust], loc='lower left', bbox_to_anchor=(.5,.5))
    ax.set_ylabel(r'Number of predictions')
    
    if save:
        name = 'misplaced_trust_cal_run_' if not disadvantage else 'disadvantage_'
        path = f"{config.ROOT_DIR}/{config.plot_path}/lenient/{name}{config.run_no_cal}_full_legend.pdf"
        create_path(f"{config.ROOT_DIR}/{config.plot_path}/lenient/")
        plt.savefig(path, bbox_inches='tight')
    # plt.show()

if __name__=='__main__':
    for show_f in [True, False]:
        alpha_vs_acc(show_full=show_f, save=True)
        
    invalid_sets_vs_misplaced_trust(disadvantage=True, save=True)
