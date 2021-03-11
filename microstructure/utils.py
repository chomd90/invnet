"""
Toy example (circle) data loader.
"""

import h5py
import torch
from torch.utils.data import Dataset


class MicrostructureDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(MicrostructureDataset, self).__init__()
        self.data = h5py.File(data_path, mode='r')['morphology_64_64']
        self.transform = transform

    def __getitem__(self, index):
        x = torch.FloatTensor(self.data[index, ...])
        if self.transform is not None:
            x = self.transform(x)
        return x/255

    def __len__(self):
        return self.data.shape[0]
