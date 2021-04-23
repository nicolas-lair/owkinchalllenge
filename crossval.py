import argparse
from functools import partial
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import random_split
from sklearn.model_selection import StratifiedKFold

from utils import get_params
from model import ChowderModel
from loader import ResNetFeaturesDataset, custom_dataloader
from trainer import train_on_one_epoch, get_model_and_optimizers, eval_model

# cuda = torch.cuda.is_available()
cuda = True

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

verbose = dict(train=False, eval=True)

if __name__ == '__main__':
    # Prepare the simulation
    args = parser.parse_args()
    filenames_train, filenames_test, labels_train, ids_test = get_params(args)
    TrainDataset = ResNetFeaturesDataset(filenames=filenames_train, labels=labels_train)
    train_auc, val_auc = dict(), dict()
    for seed in range(args.num_runs):
        train_auc[seed], val_auc[seed] = [], []

        kfold = StratifiedKFold(n_splits=args.num_splits, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(kfold.split(TrainDataset, TrainDataset.labels.numpy())):
            print(f'---- Seed, {seed}, Fold: {fold}')
            # Sample elements randomly from a list of index
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)

            # Define data loaders for training and testing in this fold
            train_loader = partial(custom_dataloader, sampler=train_subsampler)
            test_loader = partial(custom_dataloader, sampler=test_subsampler)

            chowder_model, chowder_optimizers = get_model_and_optimizers(model_type=ChowderModel, E=args.n_model,
                                                                         R=args.R, cuda=cuda)
            # Train the model on the train_index
            for e in tqdm(range(args.epoch)):
                train_on_one_epoch(chowder_model, TrainDataset, chowder_optimizers, batch_size=args.batch_size,
                                   dataloader=train_loader, verbose=verbose['train'], cuda=cuda)

            # After training, compute AUC on the train and test index
            train_auc_fold = eval_model(model=chowder_model, val_dataset=TrainDataset, dataloader=train_loader,
                                        name='train set', cuda=cuda)
            val_auc_fold = eval_model(model=chowder_model, val_dataset=TrainDataset, dataloader=test_loader,
                                      name='validation set', cuda=cuda)
            train_auc[seed].append(train_auc_fold)
            val_auc[seed].append(val_auc_fold)
    print(
        f'Average train AUC: {np.mean(list(train_auc.values())), np.std([np.mean(auc) for auc in train_auc.values()])}')
    print(f'Average val AUC: {np.mean(list(val_auc.values())), np.std([np.mean(auc) for auc in val_auc.values()])}')
