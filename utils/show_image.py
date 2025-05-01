import matplotlib.pyplot as plt

def show_image(flatten_image, dimensions, gray_scale):
    image_rgb = flatten_image.reshape((dimensions[0], dimensions[1], 1 if gray_scale else 3))
    if gray_scale:
        plt.imshow(image_rgb, cmap="gray" )
    else:
        plt.imshow(image_rgb)
    plt.title("Imagen RGB reconstruida")
    plt.show()

