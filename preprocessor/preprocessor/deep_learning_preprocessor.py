import torch
from torchvision import transforms as T

import numpy as np


def preprocess_batch(batch_images, batch_labels):
    transform = T.Compose([T.ToTensor()])
    return (
        torch.tensor(
            np.array(
                [
                    transform(image) for image in batch_images
                ]
            )
        ),
        torch.tensor(
            np.array(
                [
                    label for label in batch_labels
                ]
            )
        )       
    )
