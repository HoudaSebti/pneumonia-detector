import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, images, labels):
        self.images = images
        self.labels = labels

  def __len__(self):
        return len(self.images)

  def __getitem__(self, index):
        return self.images[index], self.labels[index]