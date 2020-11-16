import numpy as np
import pandas as pd

import cv2
from keras.utils import to_categorical
import imgaug.augmenters as iaa

from enum import Enum

import glob
import os, sys

from utilities import argument_parser

import matplotlib.pyplot as plt
import seaborn as sns

class Dataset_type(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'

def sanity_check(dataset_type):
    if not isinstance(dataset_type, Dataset_type):
        raise TypeError('the dataset type shoud be an instance of Dataset_type class !')

def generate_dataset(dataset_type, adjusted=True, data_path=None):
    sanity_check(dataset_type)
    args = argument_parser.parse_args()
    if data_path is None: 
        data_path = args.data_path
    images_paths = np.hstack(
        [
            glob.glob(
                os.path.join(
                    data_path,
                    dataset_type.value,
                    '%s/*.jpeg' %pathology
                ) 
            ) for pathology in ['normal', 'pneumonia']
        ]
    )
    dataset = pd.DataFrame(
        data={
            'images_paths' : images_paths,
            'labels' : np.array(
                [
                    0 if 'normal' in image_path else 1 for image_path in images_paths
                ]
            )
        }
    )
    if adjusted:
        return adjust_dataset(dataset)
    return dataset

def adjust_dataset(dataset, x_size=224, y_size=224, with_normalization=True):
    images_arrays = []
    images_labels = []
    for image_path, image_label in zip(dataset['images_paths'], dataset['labels']):
        image = cv2.resize(
            cv2.imread(image_path),
            (x_size, y_size)
        )
        if image.shape[2] == 1:
            image = np.dstack(
                [
                    image,
                    image,
                    image
                ]
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if with_normalization:
            image = image.astype(np.float32)/255.
        images_arrays.append(image)
        images_labels.append(
            to_categorical(image_label, num_classes=2)
        )
    return np.array(images_arrays), np.array(images_labels)

def generate_augmentation_technique(technique_name, technique_params):
    return getattr(
        sys.modules['imgaug.augmenters'],
        technique_name
    )(**technique_params)

def generate_augmentation_sequence(augmentation_techniques, aug_techs_params):
    return iaa.OneOf(
        [
            generate_augmentation_technique(
                technique_name,
                technique_params
            ) for technique_name, technique_params in zip(augmentation_techniques, aug_techs_params)
        ]
    )

def visualize_dataset_histogram(dataset_type):
    train_dataset = generate_dataset(dataset_type)
    cases_count = train_dataset['label'].value_counts()
    print(cases_count)

    plt.figure(figsize=(10,8))
    sns.barplot(x=cases_count.index, y= cases_count.values)
    plt.title('Train dataset histogram', fontsize=14)
    plt.xlabel('Case type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(len(cases_count.index)), ['normal', 'pneumonia'])
    plt.show()

if __name__ == '__main__':
    #visualize_dataset_histogram(Dataset_type.TRAIN)
    images_arrays, images_labels = generate_dataset(Dataset_type.VAL)
    print(images_arrays.shape)
    print(images_labels.shape)
    aug_seq = generate_augmentation_sequence(
        [
            'Fliplr',
            'Affine',
            'Multiply'
        ],
        [
            {},
            {
                'rotate' : 20
            },
            {
                'mul' : (1.2, 1.5)
            }
        ]
    )

    