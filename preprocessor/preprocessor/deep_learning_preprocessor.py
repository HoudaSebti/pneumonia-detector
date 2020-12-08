import torch
from torchvision import transforms as T

import numpy as np


def preprocess_batch(images, labels):
    transform = T.Compose([T.ToTensor()])
    return (
        torch.tensor(images),
        torch.tensor(labels)       
    )
