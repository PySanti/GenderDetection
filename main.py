import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras import models
from tensorflow.keras import layers
from utils.plot_precision_performance import plot_precision_performance
from utils.plot_recall_performance import plot_recall_performance
from utils.convert_images import convert_images

data_folder = "./converted_data/"
dimensions = (160,160)

[X_data, Y_data] = convert_images("./dataset/", dimensions)
print(X_data.shape)
print(Y_data.shape)
np.save("./converted_data/x_data.npy", X_data)
np.save("./converted_data/y_data.npy", Y_data)



"""
[X_data, Y_data] = [np.load(f"{data_folder}/x_data.npy"), np.load(f"{data_folder}/y_data.npy")]

X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.3, stratify=Y_data, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_val, Y_val, test_size=0.5, stratify=Y_val, random_state=42)


net = models.Sequential()
net.add(layers.Dense(1000, activation="relu", input_shape=(dimensions[0]*dimensions[1]*3,)))
net.add(layers.Dense(700, activation="relu"))
net.add(layers.Dense(500, activation="relu"))
net.add(layers.Dense(300, activation="relu"))
net.add(layers.Dense(200, activation="relu"))
net.add(layers.Dense(100, activation="relu"))
net.add(layers.Dense(50, activation="relu"))
net.add(layers.Dense(10, activation="relu"))
net.add(layers.Dense(1, activation="sigmoid"))

net.compile(loss="binary_crossentropy", optimizer="adam", metrics=["precision", "recall"])
history = net.fit(
    X_train,
    Y_train,
    epochs=25,
    validation_data=[X_val, Y_val]
)

plot_recall_performance(history)
plot_precision_performance(history)

"""
