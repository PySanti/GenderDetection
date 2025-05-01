# Gender Detection

El objetivo de este proyecto será crear una red neuronal capaz de identificar el género de una persona.

Se utilizarán dos datasets: [dataset1](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset), [dataset2](https://www.kaggle.com/datasets/ashwingupta3012/male-and-female-faces-dataset).

La arquitectura a utilizar en la red será una **fully-connected network tipo MLP**. Se utilizará `ReLu` como función de activación en las *hidden layers* y `Sigmoid` en la neurona del *output layer*, ya que es un problema de clasificación binaria.

La cantidad de neuronas, capas y épocas se irá definiendo a medida que probemos el ejercicio, ya que aún no conocemos técnicas de selección de hiperparámetros dentro de Keras.


## Preprocesamiento

El primer paso del ejercicio fue transformar las imágenes a una representación vectorial para poder ser consumidas por la red.

Para ello utilizamos las siguientes funciones:

```
import cv2
from os import listdir
from numpy.__config__ import show
from .resize_image import resize_image
import numpy as np
from .show_image import show_image

def resize_image(image, dimensions):
    image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    return image

def convert_images(folder_path, dimensions):
    X_data = []
    Y_data = []
    for gender in ["male", "female"]:
        for img in listdir(folder_path+gender):
            path = f'{folder_path}/{gender}/{img}'
            print(path)
            imagen = cv2.imread(path, cv2.IMREAD_COLOR)
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            imagen_rgb = resize_image(imagen_rgb, dimensions)
            vector = imagen_rgb.flatten().astype("float32") / 255
            target = 1 if gender=="male" else 0
            X_data.append(vector)
            Y_data.append(target)
    return [np.array(X_data), np.array(Y_data)]
```

Se cargan las imagenes, se ajustan todas a las mismas dimensiones, se aplanan y por ultimo se normalizan los valores.

Luego se cargan los datos de la siguiente manera:


```
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

data_folder = "./converted_data/"
dimensions = (160,160)

[X_data, Y_data] = [np.load(f"{data_folder}/x_data.npy"), np.load(f"{data_folder}/y_data.npy")]
X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.3, stratify=Y_data, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_val, Y_val, test_size=0.5, stratify=Y_val, random_state=42)
```


## Entrenamiento


## Evaluacion
