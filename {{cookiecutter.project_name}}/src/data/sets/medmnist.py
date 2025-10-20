import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.utils.config import instanciate_module

class MedMNISTDataset(Dataset):
    """
    PyTorch Dataset wrapper for MedMNIST datasets using dynamic instantiation.
    """
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = 'BreastMNIST',
        image_size: int = 128,
        split: str = 'train',
        transform=None
    ):
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

        self.dataset = instanciate_module(
            module_name='medmnist',
            class_name=dataset_name,
            params={
                "root": data_dir,
                "download": True,
                "transform": self.transform,
                "size": image_size,
                "split": split,
            }
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        return x, y.squeeze(0)


class SpikingMedMNISTDataset(MedMNISTDataset):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = 'BreastMNIST',
        image_size: int = 128,
        split: str = 'train',
        transform=None,
        timesteps: int = 8
    ):
        super(SpikingMedMNISTDataset, self).__init__(data_dir, dataset_name, image_size, split, transform)
        
        self.timesteps = timesteps
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Normalize((0,), (1,)),
        ])

    def encode(self, x: torch.Tensor):
        v = torch.zeros_like(x, dtype=torch.float32)
        spikes = torch.zeros((x.shape[0], self.timesteps), dtype=torch.float32, device=x.device)

        for t in range(self.timesteps):
            J = x 
            v = v + J                                
            mask = v > 1.0                           
            spikes[:, t] = mask.flatten(start_dim=1).any(dim=1).float() if v.ndim > 1 else mask.float()
            v = torch.where(mask, torch.zeros_like(v), v) 

        return spikes
        
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]    # (C, H, W)
        x = self.transform(x)       # (1, H, W)
        x = x.view(-1)              # (1*H*W)
        x = self.encode(x)          # (N, T)
        x = x.unsqueeze(0)
        return x, y[0]