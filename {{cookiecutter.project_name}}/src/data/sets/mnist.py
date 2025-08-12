import torch
import numpy as np
from lava.utils.dataloader.mnist import MnistDataset
from torch.utils.data import Dataset


class SMNIST(Dataset):
  def __init__(self, is_train, n_steps, gain=1, bias=0):
    super(SMNIST, self).__init__()
    mnist_dset = MnistDataset()
    self.n_steps = n_steps
    self.gain, self.bias = gain, bias
    if is_train:
      self.images = mnist_dset.train_images
      self.labels = np.int64(mnist_dset.train_labels)
    else:
      self.images = mnist_dset.test_images
      self.labels = np.int64(mnist_dset.test_labels)


  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    img = self.images[idx]/255
    v = np.zeros_like(img.shape)
    spikes = np.zeros((img.shape[0], self.n_steps), dtype=np.float32)
    for t in range(self.n_steps):
      J = self.gain*img + self.bias
      v = v + J
      mask = v > 1.0 
      spikes[:, t] = np.int32(mask)
      v[mask] = 0

    return spikes, self.labels[idx]