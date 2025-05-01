from os import listdir
import numpy as np
from .vectorize_image import vectorize_image
from .show_image import show_image


def convert_images(folder_path, dimensions, gray_scale):
    X_data = []
    Y_data = []
    for gender in ["male", "female"]:
        for img in listdir(folder_path+gender):
            target = 1 if gender=="male" else 0
            path = f'{folder_path}/{gender}/{img}'
            print(f"{path} : {target}")
            vector = vectorize_image(path, dimensions, gray_scale)
            if vector is None:
                print(f"Error: No se pudo cargar la imagen {path}")
            else:
                X_data.append(vector)
                Y_data.append(target)

    return [np.array(X_data), np.array(Y_data)]



