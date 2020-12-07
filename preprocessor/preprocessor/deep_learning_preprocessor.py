import torch
from torchvision import transforms as T

import numpy as np


def preprocess_batch(images_batch, labels_batch):
    transform = T.Compose([T.ToTensor()])
    return (
        torch.tensor(images_batch),
        torch.tensor(labels_batch)       
    )
