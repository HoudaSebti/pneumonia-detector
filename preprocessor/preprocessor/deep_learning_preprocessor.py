import torch
from torchvision import transforms as T


def preprocess_batch(images_batch, labels_batch):
    transform = T.Compose([T.ToTensor()])
    return (
        torch.tensor(
            np.array(
                [
                    transform(image) for image in images_batch
                ]
            )
        ),
        torch.tensor(
            np.array(
                [
                    transform(label) for label in labels_batch
                ]
            )
        )       
    )
