import unittest

import os

from preprocessor import image_visualizer
from utilities import argument_parser


class TestPreprocessorFuncs(unittest.TestCase):
    def test_path_sanity_check(self):
        try:
            image_visualizer.path_sanity_check(
                os.path.join(
                    args.data_path,
                    'train/pneumonia/person993_bacteria_2000921.jpeg'
                )
            )
        except Exception:
            self.fail("The given image path does not exist!")


if __name__ == '__main__':
    args = argument_parser.parse_args()
    unittest.main( )