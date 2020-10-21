import os

import numpy
import matplotlib.pyplot as plt

def path_sanity_check(image_path):
    if not os.path.isfile(image_path):
        raise Exception('the path: %s does not exist!' %image_path)

def visualize_image(image_path):
    path_sanity_check(image_path)
    plt.imshow(image_path)
    plt.show()