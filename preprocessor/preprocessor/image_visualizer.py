import os

from utilities import argument_parser
from preprocessor import dataset_generator

from PIL import Image
import pywt
import random
import numpy as np

import matplotlib.pyplot as plt

def path_sanity_check(image_path):
    if not os.path.isfile(image_path):
        raise Exception('the path: %s does not exist!' %image_path)

def visualize_image(image_path):
    path_sanity_check(image_path)
    image = Image.open(image_path)
    image.show(image_path)

def get_image_shape(image_path):
    image = Image.open(image_path)
    return image.size

def visualize_pneumonia_vs_normal_dwt(pneumonia_images, normal_images, figures_number, wavelet_name, level):
    titles = [' Horizontal detail','Vertical detail', 'Diagonal detail']
    pneumonia = random.sample(list(pneumonia_images), figures_number)
    normal = random.sample(list(normal_images), figures_number)
    for i in range(figures_number):
        normal_dwt = pywt.wavedec2(normal[i], wavelet_name, level=level)
        pneumonia_dwt = pywt.wavedec2(pneumonia[i], wavelet_name, level=level)
        fig = plt.figure(figsize=(12, 3))
        for j in range(level):
            images = np.stack(
                [
                    normal_dwt[level - j],
                    pneumonia_dwt[level - j]
                ],
                axis=0
            )
            for idx in range(images.shape[0]):
                for k in range(3 * idx, 3 * (idx + 1)):
                    ax = fig.add_subplot(level, 6, k + 1 + 6 * j )
                    ax.imshow(
                        images[idx][k - 3 * idx],
                        interpolation="nearest",
                        cmap=plt.cm.gray
                    )
                    ax.set_title(titles[k - 3 * idxj] + '_level_%s_%s' %(str(j + 1), 'normal' if idx == 0 else 'pneumonia'), fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
            fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    images, labels = dataset_generator.generate_dataset(
        dataset_generator.Dataset_type.TRAIN,
        224,
        224
    )
    visualize_pneumonia_vs_normal_dwt(
        images[labels == 1],
        images[labels == 0],
        10,
        'haar',
        2
    )
