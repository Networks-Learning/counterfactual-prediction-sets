from plotters.monotonicity import get_reward_valid_sets, read_rewards
import numpy as np
import config
import os
from utils import create_path
from pandas import concat

def valid_rewards_sorted():
    """Group rewards per image per prediction set
    and sort in descending order"""
    valid_rewards_df = get_reward_valid_sets(keep_worker_id=True)

    rws_no_index = valid_rewards_df.reset_index()
    columns_to_sort = ['image_name', 'set', 'reward']
    ascending = [True, True, False]
    rws_sorted = rws_no_index.sort_values(by=columns_to_sort, ascending=ascending)

    return rws_sorted

def do_permutation(sets, violations_fraction):
    """Randomly permute rewards for a given image"""
    random_variable = config.numpy_rng.random()
    if random_variable <= violations_fraction:
        new_sets = config.numpy_rng.permutation(sets)
    else:
        new_sets = sets
    return new_sets

def swap_rewards(violations_fraction=1.):
    # Rewards sorted per image, 
    # per prediction set (ascending), 
    # and reward value (descending)
    # may be required for other types of reward shuffling
    rws_sorted =  valid_rewards_sorted()

    # Read rewards for all pairs of
    # images and prediction sets
    all_rewards_df = read_rewards().drop(columns='timestamp').reset_index()

    # All rewards with duplicates for prediction sets
    # that include the try label
    all_rewards_with_duplicates = concat([all_rewards_df, rws_sorted], axis=0)

    # Remove duplicates to keep only rewards for prediction sets
    # that do not include the true label
    invalid_sets = all_rewards_with_duplicates.drop_duplicates(subset=['image_name', 'set','worker_id'], keep=False)                                                   

    # All image names
    image_names = np.unique(rws_sorted.image_name.values)
    for  image in image_names:
        image_q = rws_sorted['image_name']==image
        # Exclude singletons
        set_q = rws_sorted['set'] >=2
        
        # Get slice per image
        rewards_per_image = rws_sorted[(image_q)&(set_q)]
        
        # Reward values per prediction set
        sets = rewards_per_image['set'].values

        # Do the permutation
        new_sets = do_permutation(sets, violations_fraction)
        rws_sorted.loc[(image_q)&(set_q), 'set'] = new_sets 

    # Add predictions for prediction sets
    # that do not include the true label 
    swapped_with_invalid= concat([invalid_sets, rws_sorted], axis=0)

    # Save dataset with violations
    path_dir = f"{config.ROOT_DIR}/{config.violations_path}"
    create_path(path_dir)
    file_path = f"{path_dir}/{violations_fraction}_violations_strict.csv"
    swapped_with_invalid.to_csv(file_path, header=True, index=False)

if __name__ == '__main__':
    # Fraction of the data with violations
    p_vs = [0.3, 0.6, 1.0]
    for p_v in p_vs:
        # Create  dataset with violations
        swap_rewards(p_v)
        # Reset random seed
        config.numpy_rng = np.random.default_rng(seed=config.seed)
               