import unittest

import os

from preprocessor import image_visualizer
from utilities import argument_parser


class TestPreprocessorFuncs(unittest.TestCase):
    def test_path_canity_check(self, image_path):
        image_visualizer.path_sanity_check(image_path)

    def test_image_visualizer(self, image_path):
        image_visualizer.visualize_image(image_path)

if __name__ == '__main__':
    args = argument_parser.parse_args()
    unittest.main(
        os.path.join(
            args.data_path,
            'train/pneumonia/person993_bacteria_2921.jpeg'
        )
    )