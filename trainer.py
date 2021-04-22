import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score

from baseline import get_params
from loader import ResNetFeaturesDataset, custom_dataloader
from model import EnsembleChowder, ChowderModel

cuda = torch.cuda.is_available()
cuda = False

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path,
                    help="directory where data is stored", default=Path().absolute())
parser.add_argument("--num_runs", default=3, type=int,
                    help="Number of runs for the cross validation")
parser.add_argument("--num_splits", default=5, type=int,
                    help="Number of splits for the cross validation")
parser.add_argument("--batch_size", default=10, type=int,
                    help="Batch size")
parser.add_argument("--n_model", default=10, type=int,
                    help="Number of Chowder model in the ensemble model")
parser.add_argument("--R", default=5, type=int,
                    help="Number of positive and negative evidence")
parser.add_argument("--epoch", default=30, type=int,
                    help="Number of epochs to train")


def train_on_one_epoch(model, train_dataset, optimizer):
    for model_, optim_ in zip(model.model_list, optimizer):
        optim_.zero_grad()
        train_loader = custom_dataloader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        losses = []
        for i_batch, sample_batched in enumerate(train_loader):
            features, labels = sample_batched
            if cuda:
                features, labels = features.cuda(), labels.cuda()
            predictions = model_(features)
            loss = loss_fn(predictions, labels.float())
            loss += torch.norm(torch.cat([x.view(-1) for x in model_.projector.parameters()]), 2)
            losses.append(loss)
            loss.backward()
            optim_.step()
        print(f'mean loss on epoch : {torch.mean(torch.tensor(losses).cpu())}')


def eval_model(model, val_dataset):
    model.eval()
    val_loader = custom_dataloader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=4)
    val_features, val_labels = next(val_loader._get_iterator())
    val_predictions = model(val_features)
    auc_score = roc_auc_score(val_labels.cpu().numpy(), val_predictions.detach().cpu().numpy())
    print(f'validation AUC: {auc_score}')
    model.train()

if __name__ == '__main__':
    args = parser.parse_args()
    filenames_train, filenames_test, labels_train, ids_test = get_params(args)
    TrainDataset = ResNetFeaturesDataset(filenames=filenames_train, labels=labels_train)
    TestDataset = ResNetFeaturesDataset(filenames=filenames_test)
    train_dataset, val_dataset = random_split(TrainDataset, lengths=[int(0.8 * len(TrainDataset)),
                                                                     len(TrainDataset) - int(
                                                                         0.8 * len(TrainDataset))])

    model = EnsembleChowder(E=args.n_model, R=args.R)
    if cuda:
        model.cuda()
    optimizer = [optim.Adam(m.parameters(), lr=0.001) for m in model.model_list]
    loss_fn = nn.BCELoss()

    for e in range(args.epoch):
        print('-' * 5 + f' Epoch {e} ' + '-' * 5)
        train_on_one_epoch(model, train_dataset, optimizer)
        eval_model(model, val_dataset)
