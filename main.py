import numpy as np
from pandas.core.series import disallow_ndim_indexing
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras import models
from tensorflow.keras import layers
from utils.plot_precision_performance import plot_precision_performance
from utils.plot_recall_performance import plot_recall_performance
from utils.convert_images import convert_images

data_folder = "./converted_data/"
dimensions = (160,160)

[X_data, Y_data] = convert_images("./dataset/", dimensions, gray_scale=True)
np.save("./converted_data/x_data.npy", X_data)
np.save("./converted_data/y_data.npy", Y_data)
print(X_data.shape)
print(Y_data.shape)

"""
[X_data, Y_data] = [np.load(f"{data_folder}/x_data.npy"), np.load(f"{data_folder}/y_data.npy")]
X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.2, stratify=Y_data, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_val, Y_val, test_size=0.5, stratify=Y_val, random_state=42)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(X_val.shape)
print(Y_val.shape)


net = models.Sequential()
net.add(layers.Dense(300, activation="relu", input_shape=(dimensions[0]*dimensions[1]*3,)))
net.add(layers.Dense(200, activation="relu"))
net.add(layers.Dense(100, activation="relu"))
net.add(layers.Dense(50, activation="relu"))
net.add(layers.Dense(20, activation="relu"))
net.add(layers.Dense(10, activation="relu"))
net.add(layers.Dense(1, activation="sigmoid"))

net.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["precision", "recall"])
history = net.fit(
    X_train,
    Y_train,
    epochs=20,
    validation_data=[X_val, Y_val]
)


plot_recall_performance(history)
plot_precision_performance(history)
"""
