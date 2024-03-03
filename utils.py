import config 
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import numpy as np
import os

def true_label_per_image_df():
    model_path = f"{config.ROOT_DIR}/models/noise{config.NOISE_LEVEL}/{config.MODEL}_epoch10_preds.csv"
    model_df = pd.read_csv(model_path).set_index('image_name')
    true_labels = pd.DataFrame(model_df['category'], columns=['category'])
    return true_labels
        
def split_dataset(run_no):
    model_path = f"{config.ROOT_DIR}/models/noise{config.NOISE_LEVEL}/{config.MODEL}_epoch10_preds.csv"
    model_predictions = pd.read_csv(model_path)

    # Image names and true labels
    x = model_predictions['image_name'].to_numpy()
    y_strings = model_predictions['category'].to_numpy()

    # Map label strings to ints
    label_to_int_mapping_path = f"{config.ROOT_DIR}/models/label_to_int_mapping.json"
    with open(label_to_int_mapping_path, 'rt') as f:
        label_to_int_mapping = json.load(f)
    y = np.array([label_to_int_mapping[label] for label in y_strings])
            
    # Get the calibration set
    X_train, X_cal, y_train, y_cal = train_test_split(
        x, y, test_size=config.calibration_split, random_state=42+run_no)
     
    return X_train, X_cal, y_train, y_cal

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)