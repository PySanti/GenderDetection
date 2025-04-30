import cv2

def resize_image(image, dimensions):
    image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    return image


