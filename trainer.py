import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn

from baseline import get_params
from loader import ResNetFeaturesDataset, custom_dataloader
from model import EnsembleChowder, ChowderModel

cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path,
                    help="directory where data is stored", default=Path().absolute())
parser.add_argument("--num_runs", default=3, type=int,
                    help="Number of runs for the cross validation")
parser.add_argument("--num_splits", default=5, type=int,
                    help="Number of splits for the cross validation")
parser.add_argument("--batch_size", default=10, type=int,
                    help="Batch size")
parser.add_argument("--n_model", default=8, type=int,
                    help="Number of Chowder model in the ensemble model")
parser.add_argument("--R", default=5, type=int,
                    help="Number of positive and negative evidence")
parser.add_argument("--epoch", default=2, type=int,
                    help="Number of epochs to train")

if __name__ == '__main__':
    args = parser.parse_args()
    filenames_train, filenames_test, labels_train, ids_test = get_params(args)

    TrainDataset = ResNetFeaturesDataset(filenames=filenames_train, labels=labels_train)
    TestDataset = ResNetFeaturesDataset(filenames=filenames_test)

    model = ChowderModel(R=args.R)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    for e in range(args.epoch):
        print('-'*5 + f' Epoch {e} ' + '-'*5)
        optimizer.zero_grad()
        train_loader = custom_dataloader(dataset=TrainDataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        losses = []
        for i_batch, sample_batched in enumerate(train_loader):
            features, labels = sample_batched
            predictions = model(features)
            loss = loss_fn(predictions, labels.float())
            losses.append(loss)
            loss.backward(retain_graph=True)
            optimizer.step()
        print(torch.mean(torch.tensor(losses)))




