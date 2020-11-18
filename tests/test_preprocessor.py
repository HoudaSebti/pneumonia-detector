import unittest

import os

from preprocessor import image_visualizer, dataset_generator
from utilities import argument_parser


class TestPreprocessorFuncs(unittest.TestCase):
    def test_data_set_generator(self):
        self.assertEqual(
            dataset_generator.generate_dataset(
                dataset_generator.Dataset_type.TRAIN,
                224,
                224
            ).shape,
            (5216, 2)
        )
    def test_path_sanity_check(self):
        try:
            image_visualizer.path_sanity_check(
                os.path.join(
                    args.data_path,
                    'train/pneumonia/person9_bacteria_41.jpeg'
                )
            )
        except Exception:
            self.fail("The given image path does not exist!")


if __name__ == '__main__':
    args = argument_parser.parse_args()
    unittest.main( )
    