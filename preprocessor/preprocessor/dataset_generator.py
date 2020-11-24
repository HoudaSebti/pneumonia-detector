import numpy as np
import pandas as pd

import cv2
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
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

def generate_full_dataset(x_size=224, y_size=224, data_path=None):
    images, labels = {}, {}
    for dataset_type in [Dataset_type.TRAIN, Dataset_type.TEST, Dataset_type.VAL]:
        dataset_type_imgs, dataset_type_labels = generate_dataset(
            dataset_type,
            x_size,
            y_size,
            data_path
        )
        images[dataset_type] = dataset_type_imgs
        labels[dataset_type] = dataset_type_labels

    return images, labels

def generate_dataset(dataset_type, x_size, y_size, data_path):
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

    return (
        adjust_dataset(
            images_paths,
            x_size,
            y_size,
        ), np.array(
            [
                0 if 'normal' in image_path else 1 for image_path in images_paths
            ]
        )
    )       

def adjust_dataset(images_paths, x_size, y_size, with_normalization=True):
    images_arrays = []
    for image_path in images_paths:
        image = cv2.cvtColor(
            cv2.imread(image_path),
            cv2.COLOR_BGR2RGB
        )
        image = np.reshape(
            cv2.resize(
                image,
                (x_size, y_size)
            ),
            (1, ) + (x_size, y_size, 3)
        )
        if with_normalization : image = image.astype(np.float32)/255.
        images_arrays.append(image)
    return np.array(images_arrays)

def get_augmented_data(images_arrays, images_labels, batch_size=16, batch_number=500, **augmentation_techs):
    datagen = ImageDataGenerator(**augmentation_techs)
    return augment_images(
        images_arrays,
        images_labels,
        datagen,
        batch_size,
        batch_number
    )
    
def augment_images(images, labels, datagen, batch_size, batches_num, with_normalization=True):
    batches=0
    for image_batch, label_batch in datagen.flow(image, batch_size=batch_size):
        yield
        (
            image_batch.astype(np.float32)/255. if with_normalization else image_batch,
            label_batch
        )
        batches += 1
        if batches == batches_num:
            break

def visualize_dataset_histogram(dataset_type):
    dataset = generate_dataset(dataset_type)
    cases_count = dataset['labels'].value_counts()
    print(cases_count)

    plt.figure(figsize=(10,8))
    sns.barplot(x=cases_count.index, y= cases_count.values)
    plt.title('Train dataset histogram', fontsize=14)
    plt.xlabel('Case type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(range(len(cases_count.index)), ['normal', 'pneumonia'])
    plt.show()

if __name__ == '__main__':
    visualize_dataset_histogram(Dataset_type.TRAIN)


    