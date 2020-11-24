import numpy as np

from predictor import deep_learning_predictor, feature_extractors
from utilities import argument_parser
from preprocessor import dataset_generator

from skimage.io import imread, imshow
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import os

def svm_main(args, train_images, train_labels, test_images, test_labels, val_images, val_labels):
    augmented_data_generator = dataset_generator.get_augmented_data(
        train_images,
        train_labels,
        16,
        500,
        rotation_range=90,
        brightness_range=(0.3,0.8),
        horizontal_flip=True,
        height_shift_range=0.2,
        fill_mode='constant'
    )
    model = SVC(
            C=.01,
            kernel = 'rbf',
            gamma = .01,
            class_weight = {
                0 : train_labels.shape[0] / np.sum(train_labels) - 1.,
                1 : 1.,
            }
    )
    for batch_images, batch_labels in augmented_data_generator:
        model.fit(
            feature_extractors.extract_batch_hog_features(
                batch_images
            ),
            train_labels
        )

    return None

def deep_learning_main():
    #model=deep_learning_predictor.build_model((224,224,3))
    #model.summary()
    return None

if __name__ == '__main__':
    args = argument_parser.parse_args()
    images, labels = dataset_generator.generate_full_dataset()
    

    svm_main(
        args,
        images[dataset_generator.Dataset_type.TRAIN],
        labels[dataset_generator.Dataset_type.TRAIN],
        images[dataset_generator.Dataset_type.TEST],
        labels[dataset_generator.Dataset_type.TEST],
        images[dataset_generator.Dataset_type.VAL],
        labels[dataset_generator.Dataset_type.VAL]
    )




