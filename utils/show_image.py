import matplotlib.pyplot as plt

def show_image(flatten_image, dimensions):
    image_rgb = flatten_image.reshape((dimensions[0], dimensions[1], 3))
    plt.imshow(image_rgb)
    plt.title("Imagen RGB reconstruida")
    plt.show()

