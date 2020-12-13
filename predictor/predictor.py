import numpy as np

from predictor import deep_learning_predictor, feature_extractors, deep_learning_models
from utilities import argument_parser
from preprocessor import dataset_generator
import torchvision.models as models

from skimage.io import imread, imshow
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import torch.nn as nn
import torch.optim as optim

import pywt

import matplotlib.pyplot as plt

import os

def svm_main_with_hog(args, train_images, train_labels, test_images, test_labels, val_images, val_labels):
    augmented_data_generator = dataset_generator.get_augmented_data(
        train_images,
        train_labels,
        32,
        500,
        rotation_range=90,
        brightness_range=(0.3,0.8),
        horizontal_flip=True,
        height_shift_range=0.2,
        fill_mode='constant'
    )
    model = SVC(
            C=.1,
            kernel = 'rbf',
            gamma = .001,
            class_weight = 'balanced'
    )
    for images_batch, labels_batch in augmented_data_generator:
        model.fit(
            feature_extractors.extract_batch_hog_features(
                images_batch
            ),
            labels_batch
        )

    plot_confusion_matrix(
        model,
        feature_extractors.extract_batch_hog_features(test_images),
        test_labels
    )

    plt.show()

def svm_main_with_dwt(args, train_images, train_labels, test_images, test_labels, val_images, val_labels, wavelet_name, level, number_bins):
    augmented_data_generator = dataset_generator.get_augmented_data(
        train_images,
        train_labels,
        32,
        500,
        rotation_range=90,
        brightness_range=(0.3,0.8),
        horizontal_flip=True,
        height_shift_range=0.2,
        fill_mode='constant'
    )
    model = SVC(
            C=.1,
            kernel = 'rbf',
            gamma = .001,
            class_weight = 'balanced'
    )
    wt_histos = feature_extractors.get_batch_wt_histos(
        np.concatenate(
            (
                train_images,
                test_images
            ),
            axis=0
        ),
        wavelet_name,
        level,
        feature_extractors.Wt_direction.VERTICAL,
        number_bins
    )

    model.fit(
        wt_histos[: train_images.shape[0]],
        train_labels
    )
    plot_confusion_matrix(
        model,
        wt_histos[train_images.shape[0]:],
        test_labels
    )
    plt.show()

def wt_random_forest_main(train_images, train_labels, test_images, test_labels, wavelet_name, levels, wt_directions, number_bins):
    train_wt_histos = feature_extractors.get_batch_wt_histos(
        train_images,
        wavelet_name,
        levels,
        wt_directions,
        number_bins
    )
    clf = RandomForestClassifier(n_estimators=1000, max_depth=4, class_weight='balanced')
    clf.fit(
        feature_extractors.get_batch_wt_histos(
            train_images,
            wavelet_name,
            levels,
            wt_directions,
            number_bins
        ),
        train_labels
    )
    plot_confusion_matrix(
        clf,
        feature_extractors.get_batch_wt_histos(
            test_images,
            wavelet_name,
            levels,
            wt_directions,
            number_bins
        ),
        test_labels
    )
    plt.show()

def deep_learning_main(model, train_images, train_labels, test_images, test_labels, augment_data):
    if augment_data:
        augmented_data_generator = dataset_generator.get_augmented_data(
            train_images,
            train_labels,
            64,
            200,
            rotation_range=5,
            brightness_range=(0.3,0.8),
            horizontal_flip=True,
            height_shift_range=0.2,
            fill_mode='constant'
        )
        augmented_data_list = list(augmented_data_generator)
        train_images = np.concatenate(
            [
                train_images,
                np.array(
                    [
                        image
                        for augmented_data_batch in augmented_data_list
                        for image in augmented_data_batch[0]
                    ]
                )
            ],
            axis=0
        )
        train_labels = np.concatenate(
            [
                train_labels,
                np.array(
                    [
                        label
                        for augmented_data_batch in augmented_data_list
                        for label in augmented_data_batch[1]
                    ]
                )
            ],
            axis=0
        )

    deep_learning_predictor.predict_with_pytorch(
        model,
        train_images,
        train_labels,
        test_images,
        test_labels,
        'Adam',
        {'lr' : .0001},
        nn.CrossEntropyLoss(),
        100,
        64
    )

if __name__ == '__main__':
    args = argument_parser.parse_args()
    images, labels = dataset_generator.generate_full_dataset(x_size=224, y_size=224)   
    deep_learning_main(
        #deep_learning_models.BatchNormAlexNet(),
        models.vgg16_bn(),
        images[dataset_generator.Dataset_type.TRAIN],
        labels[dataset_generator.Dataset_type.TRAIN],
        images[dataset_generator.Dataset_type.TEST],
        labels[dataset_generator.Dataset_type.TEST],
        True
    )

