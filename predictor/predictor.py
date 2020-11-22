import numpy as np

from predictor import deep_learning_predictor, feature_extractors
from utilities import argument_parser
from preprocessor import dataset_generator

from skimage.io import imread, imshow
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import os

def svm_main(args, train_images, train_labels, test_images, test_labels, val_images, val_labels):
    feature_vectors = {}
    
    for images, key in zip([train_images, test_images, val_images], ['train', 'test', 'val']):
        feature_vectors[key] = np.array(
            [
                feature_extractors.extract_features_hog(
                    image
                ) for image in images
            ]
        )

    return feature_vectors

def deep_learning_main():
    #model=deep_learning_predictor.build_model((224,224,3))
    #model.summary()
    return None

if __name__ == '__main__':
    args = argument_parser.parse_args()
    images, labels = dataset_generator.generate_full_dataset(
        rotation_range=90,
        brightness_range=(0.3,0.8),
        horizontal_flip=True,
        height_shift_range=0.2,
        fill_mode='constant'
    )

    svm_main(
        args,
        images['train'],
        labels['train'],
        images['test'],
        labels['test'],
        images['val'],
        labels['val']
    )




