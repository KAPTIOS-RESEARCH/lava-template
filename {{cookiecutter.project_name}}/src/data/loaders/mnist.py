import torch
from torch.utils.data import Subset, DataLoader
from src.data.sets.mnist import SMNIST

class SMNISTDataloader(object):
    def __init__(self,
                 n_steps: int = 32,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 debug: bool = True):

        super(SMNISTDataloader, self).__init__()
        self.n_steps = n_steps
        self.debug = debug
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train(self):
        train_dataset = SMNIST(
            is_train=True,
            n_steps=self.n_steps,
        )

        if self.debug:
            train_dataset = Subset(train_dataset, range(self.batch_size * 2))

        dataloader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                pin_memory=True if torch.cuda.is_available() else False)
        return dataloader

    def val(self):
        val_dataset = SMNIST(
            is_train=True,
            n_steps=self.n_steps,
        )

        if self.debug:
            val_dataset = Subset(val_dataset, range(self.batch_size * 2))

        dataloader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                pin_memory=True if torch.cuda.is_available() else False)
        return dataloader

    def test(self):
        return self.val()
