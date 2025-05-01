import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

data_folder = "./converted_data/"
dimensions = (160,160)

[X_data, Y_data] = [np.load(f"{data_folder}/x_data.npy"), np.load(f"{data_folder}/y_data.npy")]
X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.3, stratify=Y_data, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_val, Y_val, test_size=0.5, stratify=Y_val, random_state=42)


