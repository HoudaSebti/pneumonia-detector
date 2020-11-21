from predictor import deep_learning_predictor, feature_extractors
from utilities import argument_parser
from preprocessor import dataset_generator

from skimage.io import imread, imshow
from sklearn.decomposition import PCA

import os

def svm_main(args, train_images, train_labels, test_images, test_labels, val_images, val_labels):
    feature_vectors = {}
    for images, key in zip([train_images, test_images, val_images], ['train', 'test', 'val']):
        feature_vectors[key], _ = [
            feature_extractors.extract_features_hog(
                image,
                orientations=8,
                pixels_per_cell=(16, 16),
                cells_per_block=(1,1),
                visualize=True,
                multichannel=False,
                feature_vector=True
            ) for image in images
        ]

    return None

def deep_learning_main():
    #model=deep_learning_predictor.build_model((224,224,3))
    #model.summary()
    return None

if __name__ == '__main__':
    args = argument_parser.parse_args()
    images, labels = dataset_generator.generate_full_dataset(
        dataset_generator.generate_augmentation_sequence(
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




