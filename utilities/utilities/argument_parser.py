
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '--data_path', default = '/home/houda/Workspace/data/pneumonia_data')
    parser.add_argument('-pretrained', '--pretrained', default = False)
    return parser.parse_args()
