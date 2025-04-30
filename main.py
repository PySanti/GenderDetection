import matplotlib.pyplot as plt
import numpy as np
from utils.convert_images import convert_images


[X_data, Y_data] = convert_images(folder_path="./dataset/", dimensions=(160,160))
