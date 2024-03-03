import config
import pandas as pd
import os
from utils import true_label_per_image_df

def save_sorted_model_predictions_csv():
    # Read model_prediction
    model_path = f"{config.ROOT_DIR}/models/noise{config.NOISE_LEVEL}/{config.MODEL}_epoch10_preds.csv"
        
    model_df = pd.read_csv(model_path)    

    # Isolate model output
    model_predictions_df = model_df.drop(['Unnamed: 0','model_name','noise_type','model_pred','correct','category','noise_level'], axis=1)

    # Sort labels per score (descending)
    model_predictions_df = model_predictions_df.set_index('image_name') 
    idx = model_predictions_df.values.argsort(axis=1)
    sorted_labels_asc = pd.DataFrame(model_predictions_df.columns.to_numpy()[idx], index=model_predictions_df.index)
    sorted_labels_desc = sorted_labels_asc[sorted_labels_asc.columns[::-1]].rename(columns={(15-i):i for i in range(16)})

    sorted_predictions_path = f"{config.ROOT_DIR}/models/noise{config.NOISE_LEVEL}/{config.MODEL}_epoch10_preds_sorted.csv"
    sorted_labels_desc.to_csv(sorted_predictions_path)

def get_idx_min_valid_non_singleton_set(for_forms=False):
    """Returns a dataframe  with each image and the index to its smallest valid prediction set."""
    # As index we refer to the highest score ranking of a label in the set, 
    # where the ranking is w.r.t. the scores of all the labels.

    # read dataframe with sorted predictions based on score.
    sorted_predictions_path = f"{config.ROOT_DIR}/models/noise{config.NOISE_LEVEL}/{config.MODEL}_epoch10_preds_sorted.csv"

    sorted_predictions_df = pd.read_csv(sorted_predictions_path).set_index('image_name')    

    # Get true label for each sample
    true_labels = true_label_per_image_df()

    # Index of the true label score in the predictions of each sample
    true_label_ranking_df = sorted_predictions_df.apply(lambda x: x.index[true_labels.loc[x.name]['category'] == x].to_list()[0], 1)
    # Return the index of the smallest valid non-singleton sets if needed for the forms in the human subject study
    index_min_valid_non_singleton_set_df = true_label_ranking_df.astype(int).map(lambda x : x+1 if (x==0 and for_forms) else x)

    # Compute the number of valid (non singleton) prediction sets for each sample
    n_sets_per_sample_df = index_min_valid_non_singleton_set_df.astype(int).map(lambda x: (config.N_LABELS - x))
    config.N_QUESTIONS = n_sets_per_sample_df.sum()
    
    return index_min_valid_non_singleton_set_df           
