from skimage.feature import hog
import pywt

from skimage import exposure
import matplotlib.pyplot as plt

import numpy as np


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

def extract_wavelet_features(images_batch, wavelet_name, level):
    #return np.array(
    #    [
    #        np.array(
    #            pywt.wavedec2(image, wavelet_name, level=level)
    #        )[1:].flatten() for image in images_batch
    #    ]
    #)

    return np.array(
            [
                np.hstack(
                    [
                        [
                            level_image_filter.flatten() for level_image_filter in level_image_filters
                        ] for level_image_filters in np.array(
                            pywt.wavedec2(image, wavelet_name, level=level)
                        )[1:]
                    ]
                ).flatten() for image in images_batch
            ]
        )


def visualize_wavelet_transforms(image, wavelet_name, level):
    titles = ['Approximation', ' Horizontal detail','Vertical detail', 'Diagonal detail']
    fig = plt.figure(figsize=(12, 3))
    for i in range(level):
        coeffs = pywt.wavedec2(image, wavelet_name, level=i + 1)
        for j, a in enumerate([coeffs[0], *coeffs[1]]):
            ax = fig.add_subplot(level, 4, j + 1 + 4 * i )
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(titles[j] + '_level_%s' %str(i + 1), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
    plt.show()

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
    