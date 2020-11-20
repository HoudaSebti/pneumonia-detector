from predictor import deep_learning_predictor, feature_extractors
from utilities import argument_parser
from preprocessor import dataset_generator

from skimage.io import imread, imshow

import os

def svm_main(args):

    feature_vector, hog_image = feature_extractors.extract_features_hog(
        input_image,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1,1),
        visualize=True,
        multichannel=False,
        feature_vector=True
    )
    feature_extractors.visualize_hog_image(input_image, hog_image, 'Hog for image')

def deep_learning_main():
    #model=deep_learning_predictor.build_model((224,224,3))
    #model.summary()
    return None

if __name__ == '__main__':
    args = argument_parser.parse_args()
    train_images, train_labels = dataset_generator.generate_dataset(
        dataset_generator.Dataset_type.TRAIN,
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
    svm_main(args)




