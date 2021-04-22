from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence


class ResNetFeaturesDataset(Dataset):
    """ResNet features dataset."""

    def __init__(self, filenames, labels=None):
        self.filenames = filenames
        if labels is None:
            # Useful for test dataset
            print('No labels provided ! This is probably a test dataset')
            self.labels = -torch.ones(len(self.filenames))
        else:
            # For train and val dataset
            self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        array = np.load(self.filenames[idx])
        # Remove position information from the array
        array = array[:, 3:]
        return torch.tensor(array).float(), self.labels[idx]


def custom_collate_fn(data):
    """
    As each sample can contain a different number of tiles, we use a custom function to collate the data in a batch.
    the pack_sequence function allows us to keep track of the number of tiles per image in the batch
    """
    features, labels = zip(*data)
    return pack_sequence(features, enforce_sorted=False), torch.tensor(labels)


custom_dataloader = partial(DataLoader, collate_fn=custom_collate_fn)
