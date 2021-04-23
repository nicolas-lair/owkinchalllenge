import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score

from utils import get_params, save_test_predictions
from loader import ResNetFeaturesDataset, custom_dataloader
from model import EnsembleModel, ChowderModel, WeldonModel

cuda = torch.cuda.is_available()
# cuda = False

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=Path,
                    help="directory where data is stored", default=Path().absolute())
parser.add_argument("--batch_size", default=10, type=int,
                    help="Batch size")
parser.add_argument("--n_model", default=10, type=int,
                    help="Number of Chowder model in the ensemble model")
parser.add_argument("--R", default=5, type=int,
                    help="Number of positive and negative evidence")
parser.add_argument("--epoch", default=30, type=int,
                    help="Number of epochs to train")


def get_model_and_optimizers(cuda=cuda, **kwargs):
    """
    Create model and optimizers
    :param cuda:
    :param kwargs: Parameter for EnsembleModel
    :return:
    """
    model = EnsembleModel(**kwargs)
    if cuda:
        model.cuda()
    # optimizers = [optim.Adam(m.parameters(), lr=0.001) for m in model.model_list]
    optimizers = [optim.Adam([{'params': m.projector.parameters(), 'weight_decay': 0.2},
                              {'params': m.mlp.parameters()}
                              ],
                             lr=0.001) for m in model.model_list]
    return model, optimizers


def train_on_one_epoch(model, train_dataset, optimizer, batch_size, dataloader=custom_dataloader, verbose=True,
                       cuda=cuda):
    """
    Train the ensemble model on one epoch by training each submodel separately.
    Different batches are used for each model
    :param model: EnsembleModel
    :param train_dataset: dataset containing the training samples and their labels
    :param optimizer: list of optimizers, one for each submodel
    :param batch_size: int
    :param dataloader: pytorch dataloader to be used
    :param verbose: boolean, print mean loss on epoch
    :param cuda: boolean, use gpu for computation
    :return: nothing, print the mean loss for each submodel
    """
    loss_fn = nn.BCELoss()
    for model_, optim_ in zip(model.model_list, optimizer):
        optim_.zero_grad()
        try:
            train_loader = dataloader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        except ValueError:
            train_loader = dataloader(dataset=train_dataset, batch_size=batch_size, num_workers=4)
        losses = []
        for _, sample_batched in enumerate(train_loader):
            features, labels = sample_batched
            if cuda:
                features, labels = features.cuda(), labels.cuda()
            predictions = model_(features)
            loss = loss_fn(predictions, labels.float())
            loss.backward()
            optim_.step()
            losses.append(loss.detach())
        if verbose:
            print(f'mean loss on epoch : {torch.mean(torch.tensor(losses).cpu())}')


def eval_model(model, val_dataset, name='validation', dataloader=custom_dataloader, verbose=True, cuda=cuda):
    """
    Evaluate the performance of the model on a dataset
    :param model: EnsembleModel
    :param val_dataset: dataset containing the validation samples along with their labels
    :param name: name of the validation (train or validation), used to print the result
    :param dataloader: pytorch dataloader to be used
    :param verbose: boolean, print mean loss on epoch
    :param cuda: boolean, use gpu for computation
    :return: AUC of the model on the dataset
    """
    if verbose:
        print(f"Evaluation on {name}")
    model.eval()
    val_loader = dataloader(dataset=val_dataset, batch_size=10, shuffle=False, num_workers=4)
    predictions, labels = [], []
    for _, sample_batched in enumerate(val_loader):
        val_features, val_labels = sample_batched
        labels.append(val_labels)
        if cuda:
            val_features = val_features.cuda()
        val_pred = model.predict(val_features).detach()
        predictions.append(val_pred)
    predictions, labels = torch.cat(predictions).cpu().numpy(), torch.cat(labels).cpu().numpy()
    auc_score = roc_auc_score(y_true=labels, y_score=predictions)
    if verbose:
        print(f'{name} AUC: {auc_score}')
    model.train()
    return auc_score


def eval_on_test(model, dataset, cuda=cuda):
    test_loader = custom_dataloader(dataset=dataset, batch_size=10, shuffle=False, num_workers=4)
    predictions = []
    for _, (features, _) in enumerate(test_loader):
        if cuda:
            features = features.cuda()
        predictions.append(model.predict(features).detach())
    predictions = torch.cat(predictions).cpu().numpy()
    return predictions


if __name__ == '__main__':
    # Prepare the simulation
    args = parser.parse_args()
    filenames_train, filenames_test, labels_train, ids_test = get_params(args)
    TrainDataset = ResNetFeaturesDataset(filenames=filenames_train, labels=labels_train)
    p = 0.95
    train_dataset, val_dataset = random_split(TrainDataset, lengths=[int(p * len(TrainDataset)),
                                                                     len(TrainDataset) - int(
                                                                         p * len(TrainDataset))])

    # Create the models
    chowder_model, chowder_optimizers = get_model_and_optimizers(model_type=ChowderModel, E=args.n_model, R=args.R)

    # Train
    max_val_auc = 0
    for e in range(args.epoch):
        print('-' * 5 + f' Epoch {e} ' + '-' * 5)
        print('-- Training')
        train_on_one_epoch(chowder_model, train_dataset, chowder_optimizers, batch_size=args.batch_size)
        print(' -- Evaluation')
        eval_model(chowder_model, train_dataset, name='train set')
        val_auc = eval_model(chowder_model, val_dataset, name='validation set')

        if val_auc > max_val_auc:
            # Compute predictions on the test set
            print('-- Computing the predictions on the test set')
            TestDataset = ResNetFeaturesDataset(filenames=filenames_test)
            predictions = eval_on_test(model=chowder_model, dataset=TestDataset)
            max_val_auc = val_auc


    print(f'Predictions on the test were computed for AUC on val of {max_val_auc}')
    save_test_predictions(ids_test=ids_test, predictions=predictions, data_dir=args.data_dir)
