import os

from utilities import argument_parser
from preprocessor import dataset_generator

from PIL import Image
import pywt

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
    pneumonia = random.sample(pneumonia_images, figures_number)
    normal = random.sample(normal_images, figures_number)
    for i in range(figures_number):
        normal_dwt = pywt.wavedec2(normal[i], wavelet_name, level=level)
        pneumonia_dwt = pywt.wavedec2(normal[i], wavelet_name, level=level)
        fig = plt.figure(figsize=(12, 3))

        for j in range(level):
            images = np.stack(
                [
                    normal_dwt[level - j],
                    pneumonia_dwt[level - j]
                ]
            )
            for idx in range(images.shape[0]):
                for k in range(3 * idx, 3 * (idx + 1)):
                    ax = fig.add_subplot(level, 6, k + 1 + 6 * j )
                    ax.imshow(
                        images[idx][k],
                        interpolation="nearest",
                        cmap=plt.cm.gray
                    )
    plt.show()

if __name__ == '__main__':
    images, labels = dataset_generator.generate_dataset(
        dataset_generator.TRAIN,
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
