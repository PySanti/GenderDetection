from os import listdir
from .resize_image import resize_image
import cv2
import numpy as np


def convert_images(folder_path, dimensions):
    X_data = []
    Y_data = []
    for gender in ["male", "female"]:
        for img in listdir(folder_path+gender):
            target = 1 if gender=="male" else 0
            path = f'{folder_path}/{gender}/{img}'
            print(f"{path} : {target}")
            imagen = cv2.imread(path, cv2.IMREAD_COLOR)
            if imagen is None:
                print(f"Error: No se pudo cargar la imagen {path}")
                continue  # Salta la imagen problemática
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            imagen_rgb = resize_image(imagen_rgb, dimensions)
            vector = imagen_rgb.flatten().astype("float32") / 255
            X_data.append(vector)
            Y_data.append(target)
    return [np.array(X_data), np.array(Y_data)]



