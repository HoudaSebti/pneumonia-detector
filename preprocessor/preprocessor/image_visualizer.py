import os

from utilities import argument_parser

from PIL import Image

def path_sanity_check(image_path):
    if not os.path.isfile(image_path):
        raise Exception('the path: %s does not exist!' %image_path)

def visualize_image(image_path):
    path_sanity_check(image_path)
    image = Image.open(image_path)
    image.show(image_path)

def get_image_shape(image_path):
    image = Image.open(image_path)
    return image.size

if __name__ == '__main__':
    args = argument_parser.parse_args()
    print(
        get_image_shape(
            args.data_path
        )
    )
    visualize_image(
        os.path.join(
            args.data_path,
            'train/normal/NORMAL2-IM-0856-0001.jpeg'
        )
    )
