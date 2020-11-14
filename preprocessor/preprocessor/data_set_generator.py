import numpy as np
import pandas as pd

from enum import Enum

import glob

from utilities import argument_parser

class Dataset_type(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'

def sanity_check(data_set_type):
    if not isinstance(data_set_type, Dataset_type):
        raise TypeError('the dataset type shoud be an instance od Fataset_type class !')

def generate_dataset(data_set_type, data_path=None):
    sanity_check(data_set_type)
    args = argument_parser.parse_args()
    if data_path is None: 
        data_path = args.data_path
    images_paths = np.vstack(
        [
            glob.glob(
                os.path.join(
                    data_path,
                    data_set_type.value,
                    '%s/*.jpeg' %pathology
                ) 
            ) for pathology in ['normal', 'pneumonia']
        ]
    )
    return pd.Dataframe(
        data={
            'image_path' : images_paths,
            'label' : np.array(
                [
                    0 if 'normal' in image_path else 1 for image_path in images_paths
                ]
            )
        }
    )
