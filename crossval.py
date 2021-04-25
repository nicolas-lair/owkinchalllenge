from functools import partial

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import random_split
from sklearn.model_selection import StratifiedKFold

from utils import get_params, get_model_and_optimizer, train_on_one_epoch, eval_model
from loader import ResNetFeaturesDataset, custom_dataloader
from config import Chowder_CONFIG, multiR_CONFIG, DeepSet_CONFIG

config = Chowder_CONFIG

verbose = dict(train=False, eval=True)

if __name__ == '__main__':
    print(Chowder_CONFIG)
    # Prepare the simulation
    filenames_train, filenames_test, labels_train, ids_test = get_params(config)
    TrainDataset = ResNetFeaturesDataset(filenames=filenames_train, labels=labels_train)
    train_auc, val_auc = dict(), dict()
    for seed in range(config.num_runs):
        train_auc[seed], val_auc[seed] = [], []

        kfold = StratifiedKFold(n_splits=config.num_splits, shuffle=True, random_state=seed)
        for fold, (train_index, test_index) in enumerate(kfold.split(TrainDataset, TrainDataset.labels.numpy())):
            val_auc[seed].append([])
            print(f'---- Seed, {seed}, Fold: {fold}')
            # Sample elements randomly from a list of index
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)

            # Define data loaders for training and testing in this fold
            train_loader = partial(custom_dataloader, sampler=train_subsampler)
            test_loader = partial(custom_dataloader, sampler=test_subsampler)

            model, optimizers = get_model_and_optimizer(mtype=config.mtype, model_params=config.model_params)
            # Train the model on the train_index
            for e in tqdm(range(config.epoch)):
                train_on_one_epoch(model=model, train_dataset=TrainDataset, optimizer=optimizers,
                                   dataloader=train_loader, verbose=verbose['train'])
                # Eval on the validation set after each epoch of training
                val_auc[seed][fold].append(eval_model(model=model, val_dataset=TrainDataset, dataloader=test_loader,
                                                      name='validation set', verbose=False))

            # After training, compute AUC on the train index
            train_auc_fold = eval_model(model=model, val_dataset=TrainDataset, dataloader=train_loader,
                                        name='train set')
            train_auc[seed].append(train_auc_fold)

            # Format the validation auc as a pd.Series
            val_auc[seed][fold] = pd.Series(val_auc[seed][fold], name=fold)
            print(val_auc[seed][fold])

        # Aggregate the validation auc of a complete CV as a pd.DataFrame
        val_auc[seed] = pd.DataFrame(val_auc[seed]).T.stack()

    # Aggregate the validation auc of the complete simulation
    val_auc = pd.DataFrame(val_auc).T.stack().stack()
    val_auc.index.rename(["seed", 'fold', 'epoch'], inplace=True)

    # Compute mean auc and std auc
    mean_val = val_auc.groupby('epoch').mean().rename('mean', axis=1)
    std_val = val_auc.groupby(['seed', 'epoch']).mean().groupby('epoch').std().rename('std', axis=1)

    print(
        f'Average train AUC: {np.mean(list(train_auc.values())), np.std([np.mean(auc) for auc in train_auc.values()])}')
    print(f'Average val AUC: {pd.concat([mean_val, std_val], axis=1).sort_values(by="mean")}')
