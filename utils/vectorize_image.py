from .resize_image import resize_image
import cv2

def vectorize_image(path, dimensions, gray_scale):
    imagen = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray_scale else cv2.IMREAD_COLOR)
    if imagen is None:
        return None
    if not gray_scale:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_rgb = resize_image(imagen, dimensions)
    vector = imagen_rgb.flatten().astype("float32") / 255
    return vector
