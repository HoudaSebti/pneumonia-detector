import os

import numpy

from utilities import argument_parser

import matplotlib.pyplot as plt
from PIL import Image

def path_sanity_check(image_path):
    if not os.path.isfile(image_path):
        raise Exception('the path: %s does not exist!' %image_path)

def visualize_image(image_path):
    path_sanity_check(image_path)
    plt.imshow(image_path)
    plt.show()

def get_image_shape(image_path):
    image = Image.open(image_path)
    return image.size

if __name__ == '__main__':
    args = argument_parser.parse_args()
    print(
        get_image_shape(
            args.data_path
        )
    )
    visualize_image(args.data_path)
