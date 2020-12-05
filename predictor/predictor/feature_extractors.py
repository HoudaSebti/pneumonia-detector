from skimage.feature import hog
import pywt

from skimage import exposure
import matplotlib.pyplot as plt

import numpy as np
import random
import math
from enum import Enum

from preprocessor import dataset_generator

class Wt_direction(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    DIAGONAL = 2


def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3,3),visualize=False, multichannel=False, feature_vector=True):
    if len(image.shape) == 3 and image.shape[-1] == 1:
        image = image.reshape((image.shape[0], image.shape[1]))
    return hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=visualize,
        multichannel=multichannel,
        feature_vector=feature_vector
    )

def extract_batch_hog_features(batch_images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),visualize=False, multichannel=False, feature_vector=True):
    return np.array(
        [
            extract_hog_features(
                image,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=visualize,
                multichannel=multichannel,
                feature_vector=feature_vector
            ) for image in batch_images
        ]
    )

def get_batch_wavelet_histogram(images_batch, wavelet_name, level, wt_direction, bins_number, just_histo=True):
    wavelet_transforms = np.array(
        [
            get_wavelet_transform(
                image,
                wavelet_name,
                level,
                wt_direction
            ).flatten() for image in images_batch
        ]
    )
    #histo_min = np.min(wavelet_transforms)
    #histo_max = np.max(wavelet_transforms)
    histo_min=np.percentile(wavelet_transforms, 1)
    histo_max=np.percentile(wavelet_transforms, 99)

    return np.array(
        [
            get_wavelet_trans_histogram(
                wavelet_transform,
                histo_max,
                histo_min,
                bins_number,
                just_histo
            ) for wavelet_transform in wavelet_transforms
        ]
    )
    

def get_wavelet_trans_histogram(image, histo_max, histo_min, bins_number, just_histo):
    step = (histo_max - histo_min) / bins_number
    histo = np.histogram(
        image,
        bins=np.arange(
            histo_min,
            histo_max + step,
            step
        )   
    )
    return histo[0] if just_histo else histo

def get_wavelet_transform(image, wavelet_name, level, wt_direction):
    if not isinstance(wt_direction, Wt_direction):
        raise NotImplementedError('the wt_direction argument should an instance of class: feature_extractors.Wt_direction')
    return pywt.wavedec2(image, wavelet_name, level=level)[1][wt_direction.value]

def visualize_hog_image(input_image, hog_image, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(input_image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title(title)


def wavelet_trans_histos_main():
    figures_number = 20
    train_images, train_labels = dataset_generator.generate_dataset(dataset_generator.Dataset_type.TRAIN,224,224)
    test_images, test_labels = dataset_generator.generate_dataset(dataset_generator.Dataset_type.TRAIN,224,224)
    images = np.concatenate(
        [
            train_images,
            test_images
        ],
        axis=0
    )
    labels = np.concatenate(
        [
            train_labels, test_labels
        ],
        axis=0
    )
    histos = get_batch_wavelet_histogram(
        images,
        'haar',
        1,
        Wt_direction.VERTICAL,
        50,
        False
    )
    pneumonia_indices = random.sample(
        list(range(images[labels==1].shape[0])),
        figures_number
    )
    normal_indices = random.sample(
        list(range(images[labels==0].shape[0])),
        figures_number
    )
    for pneumonia_idx, normal_idx in zip(pneumonia_indices, normal_indices):
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(1, 2, 1)
        ax.bar(histos[normal_idx, 1][:-1], height=histos[normal_idx, 0], align='edge')
        ax.set_title('normal histogram')
        ax = fig.add_subplot(1, 2, 2)
        ax.bar(histos[pneumonia_idx, 1][:-1], height=histos[pneumonia_idx, 0], align='edge')
        ax.set_title('pneumonia histogram')
    plt.show()

if __name__ == '__main__':
    wavelet_trans_histos_main()

