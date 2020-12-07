import torch
from torchvision import transforms as T


def preprocess_batch(images_batch):
    transform = T.Compose([T.ToTensor()])
    return [transform(image) for image in images_batch]
