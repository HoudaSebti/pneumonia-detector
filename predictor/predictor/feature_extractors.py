from skimage.feature import hog

from skimage import exposure
import matplotlib.pyplot as plt


def extract_features_hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1,1),visualize=True, multichannel=True, feature_vector=True):
    return hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=visualize,
        multichannel=multichannel,
        feature_vector=feature_vector
    )

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
    plt.show()