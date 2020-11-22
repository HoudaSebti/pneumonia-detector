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

def generate_full_dataset(**augmentation_techs=None):
    images, labels = {}, {}
    for dataset_type in [Dataset_type.TRAIN, Dataset_type.TEST, Dataset_type.VAL]:
        dataset_type_imgs, dataset_type_labels = generate_dataset(
            dataset_type,
            augmentation_techs
        )
        images[dataset_type.value] = dataset_type_imgs
        labels[dataset_type.value] = dataset_type_labels

    return images, labels

def generate_dataset(dataset_type, augmentation_techs=None, x_size=224, y_size=224, data_path=None):
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
    return adjust_dataset(
        pd.DataFrame(
            data={
                'images_paths' : images_paths,
                'labels' : np.array(
                    [
                        0 if 'normal' in image_path else 1 for image_path in images_paths
                    ]
                )
            }
        ),
        x_size,
        y_size,
        augmentation_techs=augmentation_techs
    )

def adjust_dataset(dataset, x_size, y_size, augmentation_techs=None, with_normalization=True):
    images_arrays, images_labels = [], []
    cases_count = dataset['labels'].value_counts().to_dict()
    for image_path, image_label in zip(dataset['images_paths'], dataset['labels']):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.reshape(
            cv2.resize(
                cv2.imread(image_path),
                (x_size, y_size)
            ),
            (1, ) + (x_size, y_size, 3)
        )
        if with_normalization : image = image.astype(np.float32)/255.
        if augmentation_techs is not None:
            datagen = ImageDataGenerator(**augmentation_techs)
            new_images, new_labels = augment_image(
                image,
                image_label,
                datagen
            )
        else:
            new_images, new_labels = [image], [image_label]
        update_data(
            images_arrays,
            images_labels,
            new_images,
            new_labels
        )
    return np.array(images_arrays), np.array(images_labels)

def update_data(images_arrays, images_labels, new_images, new_labels):
    images_arrays.extend(new_images)
    images_labels.extend(
        [
            #to_categorical(
            #    image_label,
            #    num_classes=2
            #) for image_label in new_labels
            new_labels
        ]
    )
    
def generate_augmentation_technique(technique_name, technique_params):
    return getattr(
        sys.modules['imgaug.augmenters'],
        technique_name
    )(**technique_params)

def in_minority(label, cases_count):
    return cases_count[label] < cases_count[1 - label]

def get_augmentation_number(cases_count):
    count_vals = list(cases_count.values())
    return int(
        np.max(count_vals) / np.min(count_vals)
    )

def augment_image(image, label, datagen, batch_size=16, batches_num=10, with_normalization=True):
    batches=0
    images, labels = [image], [label]
    for image_batch, label_batch in datagen.flow(image, batch_size=batch_size):
        images.append(image_batch.astype(np.float32)/255. if with_normalization else image_batch)
        labels.append(label_batch.astype(np.float32)/255. if with_normalization else label_batch)
        batches += 1
        if batches > batches_num:
            break
    return (
        images,
        labels
    )

def generate_augmentation_sequence(augmentation_techniques, aug_techs_params):
    return iaa.OneOf(
        [
            generate_augmentation_technique(
                technique_name,
                technique_params
            ) for technique_name, technique_params in zip(augmentation_techniques, aug_techs_params)
        ]
    )

def get_batches_generator(images, labels, batch_size=16):
    indices = np.random.choice(
        range(images.shape[0]),
        batch_size
    )
    for i in range(images.shape[0] // batch_size):
        print('generating batch number: %s' %str(i))
        yield (
            images[indices],
            labels[indices]
        )

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


    