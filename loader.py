from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence


class ResNetFeaturesDataset(Dataset):
    """ResNet features dataset."""

    def __init__(self, filenames, labels=None):
        self.ids = {f.stem: f for f in filenames}
        if labels is None:
            print('No labels provided ! This is probably a test dataset')
            self.labels = -torch.ones(len(self.ids))
        else:
            self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            ids = list(self.ids)[idx]
        # elif isinstance(idx, str):
        #     ids = idx
        #     idx = list(self.ids).index(idx)
        else:
            raise NotImplementedError

        array = np.load(self.ids[ids])
        array = array[:, 3:]
        return torch.tensor(array).float(), self.labels[idx]


def custom_collate_fn(data):
    features, labels = zip(*data)
    # return pad_sequence(features, batch_first=True), torch.tensor(labels)
    return pack_sequence(features, enforce_sorted=False), torch.tensor(labels)


custom_dataloader = partial(DataLoader, collate_fn=custom_collate_fn)
