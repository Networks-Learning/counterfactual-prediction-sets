import config
import pandas as pd
import numpy as np
from utils import split_dataset
class Model:
    def __init__(self) -> None:
        model_pred_path = f"{config.ROOT_DIR}/models/noise{config.NOISE_LEVEL}/{config.MODEL}_epoch10_preds.csv"
        self.preds = pd.read_csv(model_pred_path).set_index('image_name')
        
    def accuracy(self, images=None):
        """Average accuracy of the model over all (or a set of) images"""
        if all(images)!= None:
            model_acc = self.preds.loc[images, 'correct'].mean()
        else:
            model_acc = self.preds['correct'].mean()

        return model_acc

    def acc_se(self, images=None):
        """Standard error of the accuracy of the model"""
        if all(images)!=None:
            model_se = self.preds.loc[images, 'correct'].std() / np.sqrt(len(self.preds.loc[images, 'correct']))
        else:
            model_se = self.preds['correct'].std() / np.sqrt(len(self.preds['correct']))

        return model_se

    def pred_prob(self, x):
        """The model predicted probabilities"""
        pred_probs = self.preds.loc[x, 'knife':'dog'].to_numpy()
        return pred_probs

    def pred_prob_sorted(self, x):
        pred_probs = self.pred_prob(x)
        sorted_desc = -np.sort(-pred_probs)
        return sorted_desc

if __name__=="__main__":
    X_train, X_cal, y_train, y_cal = split_dataset(config.run_no_cal)

    # Shuffle train set
    config.numpy_rng.shuffle(X_train)

    config.total_timesteps = len(X_train)
    # Create model
    model = Model()
    print(model.accuracy(X_train))
    print(model.acc_se(X_train))
    