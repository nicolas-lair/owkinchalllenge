import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score

from utils import get_params
from loader import ResNetFeaturesDataset, custom_dataloader
from model import EnsembleModel, ChowderModel, WeldonModel

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

weldon = False


def get_model_and_optimizers(**kwargs):
    """
    Create model and optimizers
    :param kwargs: Parameter for EnsembleModel
    :return:
    """
    model = EnsembleModel(**kwargs)
    if cuda:
        model.cuda()
    optimizers = [optim.Adam(m.parameters(), lr=0.001) for m in model.model_list]
    # optimizers = [optim.Adam([{'params': m.projector.parameters(), 'weight_decay': 0.5},
    #                           {'params': m.mlp.parameters()}
    #                           ],
    #                          lr=0.001) for m in model.model_list]
    return model, optimizers


def train_on_one_epoch(model, train_dataset, optimizer):
    """
    Train the ensemble model on one epoch by training each submodel separately.
    Different batches are used for each model
    :param model: EnsembleModel
    :param train_dataset: dataset containing the training samples and their labels
    :param optimizer: list of optimizers, one for each submodel
    :return: nothing, print the mean loss for each submodel
    """
    for model_, optim_ in zip(model.model_list, optimizer):
        optim_.zero_grad()
        train_loader = custom_dataloader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        losses = []
        for _, sample_batched in enumerate(train_loader):
            features, labels = sample_batched
            if cuda:
                features, labels = features.cuda(), labels.cuda()
            predictions = model_(features)
            loss = loss_fn(predictions, labels.float())
            # loss += 0.5 * torch.norm(torch.cat([x.view(-1) for x in model_.projector.parameters()]), 2)
            loss.backward()
            optim_.step()
            losses.append(loss.detach())
        print(f'mean loss on epoch : {torch.mean(torch.tensor(losses).cpu())}')


def eval_model(model, val_dataset, name='validation'):
    """
    Evaluate the performance of the model on a dataset
    :param model: EnsembleModel
    :param val_dataset: dataset containing the validation samples along with their labels
    :param name: name of the validation (train or validation), used to print the result
    :return: nothing, print the AUC of the model on the dataset
    """
    model.eval()
    val_loader = custom_dataloader(dataset=val_dataset, batch_size=10, shuffle=False, num_workers=4)
    predictions, labels = [], []
    for _, sample_batched in enumerate(val_loader):
        val_features, val_labels = sample_batched
        labels.append(val_labels)
        val_pred = model.predict(val_features).detach()
        predictions.append(val_pred)
    predictions, labels = torch.cat(predictions).cpu().numpy(), torch.cat(labels).cpu().numpy()
    auc_score = roc_auc_score(y_true=labels, y_score=predictions)
    print(f'{name} AUC: {auc_score}')
    model.train()


if __name__ == '__main__':
    # Prepare the simulation
    args = parser.parse_args()
    filenames_train, filenames_test, labels_train, ids_test = get_params(args)
    TrainDataset = ResNetFeaturesDataset(filenames=filenames_train, labels=labels_train)
    # TestDataset = ResNetFeaturesDataset(filenames=filenames_test)
    train_dataset, val_dataset = random_split(TrainDataset, lengths=[int(0.8 * len(TrainDataset)),
                                                                     len(TrainDataset) - int(
                                                                         0.8 * len(TrainDataset))])

    # Create the models
    chowder_model, chowder_optimizers = get_model_and_optimizers(model_type=ChowderModel, E=args.n_model, R=args.R)
    if weldon:
        weldon_model, weldon_optimizers = get_model_and_optimizers(model_type=WeldonModel, E=args.n_model, R=args.R)

    loss_fn = nn.BCELoss()

    for e in range(args.epoch):
        print('-' * 5 + f' Epoch {e} ' + '-' * 5)

        print('-- Training')
        print('---- Chowder')
        train_on_one_epoch(chowder_model, train_dataset, chowder_optimizers)
        if weldon:
            print('---- Weldon')
            train_on_one_epoch(weldon_model, train_dataset, weldon_optimizers)

        print('-- Evaluation of Chowder')
        eval_model(chowder_model, train_dataset, name='train')
        eval_model(chowder_model, val_dataset, name='validation')
        if weldon:
            print('-- Evaluation of Weldon')
            eval_model(weldon_model, train_dataset, name='train')
            eval_model(weldon_model, val_dataset, name='validation')
