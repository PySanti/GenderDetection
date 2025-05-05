import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from utils.plot_precision_performance import plot_precision_performance
from utils.plot_recall_performance import plot_recall_performance
from tensorflow.keras import optimizers

data_folder = "./converted_data/"
dimensions = (160,160)

[X_data, Y_data] = [np.load(f"{data_folder}/x_data.npy"), np.load(f"{data_folder}/y_data.npy")]

X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.20, stratify=Y_data, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_val, Y_val, test_size=0.5, stratify=Y_val, random_state=42)

net = models.Sequential()
net.add(layers.Dense(500,  activation="relu", input_shape=(dimensions[0]*dimensions[1]*3,)))
net.add(layers.Dense(300, activation="relu"))
net.add(layers.Dense(200, activation="relu"))
net.add(layers.Dense(100, activation="relu"))
net.add(layers.Dense(50, activation="relu"))
net.add(layers.Dense(20, activation="relu"))
net.add(layers.Dense(1, activation="sigmoid"))

net.compile(loss="binary_crossentropy", optimizer=optimizers.SGD(learning_rate=0.0001), metrics=["precision", "recall"])
history = net.fit(
    X_train,
    Y_train,
    epochs=30,
    validation_data=[X_val, Y_val]
)


plot_recall_performance(history)
plot_precision_performance(history)

