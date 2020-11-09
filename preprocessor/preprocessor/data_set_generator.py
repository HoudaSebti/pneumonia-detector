import numpy as np
import pandas as pd

from enum import Enum

from utilities import argument_parser

class Dataset_type(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'

def generate_dataset(data_set_type, data_path=None):
    args = argument_parser.parse_args()
    if data_path is None: 
        data_path = args.data_path
