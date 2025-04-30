from os import listdir
from .resize_image import resize_image
import cv2


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
    return [X_data, Y_data]



