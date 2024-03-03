import config 
import pandas as pd
from models.preprocess_predictions import get_idx_min_valid_non_singleton_set as pivot_sets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import create_path

mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts,geometry}'
mpl.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams.update({
    'font.family':'serif',
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "font.size": 48,
    "figure.figsize":(16,10)
})

Y_AXIS_BORDERS_VIOALTIONS = {
    1: (0.25, 0.6),
    2: (0.65, 0.85),
    3: (0.89, 0.96)
}
def read_rewards(setting='strict'):
    """Returns the rewards for each image and prediction set"""
    rewards_path = f"{config.ROOT_DIR}/{config.rewards_file}_{setting}.csv"
    rewards = pd.read_csv(rewards_path, dtype={'set':np.int32, 'reward':np.bool_}).set_index('image_name')
    return rewards

def get_pivot_sets():
    """Returns the smallest set that includes the true label
    for each image"""
    pivot_s_df = pivot_sets().to_frame()
    pivot_s_df.rename(columns={0:'pivot_set'}, inplace=True)
    pivot_s_df+=1
    return pivot_s_df

def get_reward_valid_sets(setting='strict',keep_worker_id=False):
    """Returns the rewards for the prediction sets 
    that include the true label"""
    rewards = read_rewards(setting=setting)
    columns_to_keep = ['set', 'reward']
    if keep_worker_id:
        columns_to_keep.append('worker_id')
    if setting=='lenient':
        columns_to_keep.append('is_other')
    set_reward_df = rewards[columns_to_keep]

    pivot_s_df = get_pivot_sets()
    set_reward_pivot = set_reward_df.join(other=pivot_s_df, how='left')

    reward_valid_sets = set_reward_pivot[set_reward_pivot['set'] >= set_reward_pivot['pivot_set']][columns_to_keep]
    return reward_valid_sets

def plot(reward_valid_sets, ylim_low=None, ylim_high=None, n_image_strata=0):
    """Plots the empirical success probability per prediction set size
    for the prediction sets that include the true label"""
    errors = (reward_valid_sets.groupby('set').std() / np.sqrt(reward_valid_sets.groupby('set').count()))
    acc_per_set = reward_valid_sets.groupby('set').mean()
    print(acc_per_set)
    ax = acc_per_set.plot.bar(yerr=errors, rot=0)
    low_lim = .02 if n_image_strata < 3 else .002
    high_lim = 0.01 if n_image_strata < 3 else .001
    ylim_low = min(acc_per_set.values) - errors.values[np.argmin(acc_per_set.values)] - low_lim if not ylim_low else ylim_low
    ylim_high = max(acc_per_set.values) + high_lim if not ylim_high else ylim_high
    # Fix y axis for difficulty levels 1 to 3  
    # for datasets with violations 
    if config.violations > 0 and n_image_strata < 4:
        ylim_low = Y_AXIS_BORDERS_VIOALTIONS[n_image_strata][0]
        ylim_high = Y_AXIS_BORDERS_VIOALTIONS[n_image_strata][1]
    ax.set_ylim(bottom=ylim_low, top=ylim_high)
    ax.set_xlabel(r"Prediction Set Size")
    ax.set_ylabel(r"Empirical Success Probability")
    ax.tick_params(axis='y', width=4, length=20)
    ax.spines[['right', 'top']].set_visible(False)
    ax.get_legend().remove()

def thresholds_per_strata(key, reward_valid_sets, n_stratas):    
    """Compute the threshold value for each strata of <key>,
    where <key> can be one of {'image_name', 'worker_id'}"""
    avg_acc_per_key = reward_valid_sets.groupby(key).mean(numeric_only=True)['reward'].to_frame()
    acc_thresholds_per_strata = {}
    for i in range(1, n_stratas+1):        
        acc_thresholds_per_strata[i] = {
            "high": avg_acc_per_key.quantile(i/n_stratas, axis=0)['reward']
        }

    return acc_thresholds_per_strata, avg_acc_per_key

def get_strata(n_strata, low_threshold, avg_acc_per_key, acc_thr_per_strata, is_last=False):
    """Returns the strata of <key>, that is images of similar 
    difficulty or workers with the same level of competence"""
    lb_q = avg_acc_per_key['reward'] >= low_threshold
    ub_q = avg_acc_per_key['reward'] < acc_thr_per_strata[n_strata]["high"]
    if is_last:
        ub_q = avg_acc_per_key['reward'] <= acc_thr_per_strata[n_strata]["high"]

    strata = avg_acc_per_key[(lb_q)&(ub_q)].index
    return strata

def per_strata(n_stratas_workers=1, n_stratas_images=1, strata_to_plot_workers=0, strata_to_plot_images=0):
    """Stratify dataset with respect to the difficulty of the images 
    and the competence level of workers and plot the empirical success
    probability per prediction set size for prediction sets that include the 
    true label for each strata"""
    def plot_strata(reward_valid_sets, worker_strata_n, image_strata_n, workers_total_stratas, images_total_stratas):
        plot(reward_valid_sets, n_image_strata=image_strata_n)
        worker_dir_path = f"{config.ROOT_DIR}/{config.plot_path}/monotonicity/workers{workers_total_stratas}/"
        create_path(worker_dir_path)

        worker_image_path = worker_dir_path + f"images{images_total_stratas}/"
        if not os.path.exists(worker_image_path):
            os.mkdir(worker_image_path)

        plot_path = worker_image_path + f"workers_{worker_strata_n}_images_{image_strata_n}_nosingleton.pdf"

        plt.savefig(plot_path, bbox_inches='tight')
        
    reward_valid_sets = get_reward_valid_sets(keep_worker_id=True)
    workers_acc_thr_per_strata, avg_acc_per_worker = thresholds_per_strata('worker_id', reward_valid_sets, n_stratas_workers)
    
    images_acc_thr_per_strata, avg_acc_per_image = thresholds_per_strata('image_name', reward_valid_sets, n_stratas_images)
    
    workers_low_threshold =  0
    for i in range(1, n_stratas_workers+1):
        # Get workers strata
        strata_workers = get_strata(i, workers_low_threshold, avg_acc_per_worker, workers_acc_thr_per_strata)
        
        images_low_threshold = 0
        for j in range(1, n_stratas_images+1):
            is_last = j == n_stratas_images
            # Get images strata
            strata_images = get_strata(j, images_low_threshold, avg_acc_per_image, images_acc_thr_per_strata, is_last)

            workers_q = reward_valid_sets['worker_id'].isin(strata_workers)
            images_q = reward_valid_sets.index.isin(strata_images)
            
            strata = reward_valid_sets.loc[(workers_q)&(images_q)][['set', 'reward']]
            strata = strata.astype({'set': 'int16'})
            plot_all = strata_to_plot_workers == 0 and strata_to_plot_images == 0
            plot_this_strata = strata_to_plot_workers == i and strata_to_plot_images == j
            if plot_all or plot_this_strata:
                plot_strata(strata, worker_strata_n=i, image_strata_n=j, workers_total_stratas=n_stratas_workers, images_total_stratas=n_stratas_images)
        
            images_low_threshold = images_acc_thr_per_strata[j]["high"]

        workers_low_threshold = workers_acc_thr_per_strata[i]["high"]

if __name__=="__main__":
    if config.violations == 0:
        # Human subject study dataset 
        # 5 difficulty levels for images, one competence level for experts (all experts)
        per_strata(n_stratas_images=5, n_stratas_workers=1)
        # 5 difficulty levels for images, two competence levels for experts
        per_strata(n_stratas_images=5, n_stratas_workers=2)
    else:
        # Violations of interventional monotonicity
        # 5 difficulty levels for images, one competence level for experts (all experts)
        per_strata(n_stratas_images=5, n_stratas_workers=1)
    