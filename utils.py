import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import optim as optim, nn as nn

from loader import custom_dataloader
from model import EnsembleModel, ChowderModel
from config import baseCONFIG


def save_test_predictions(ids_test, predictions, data_dir):
    """
    Piece of code taken from the baselines for usage in the trainer file
    Save the test predictions in the right format for submission on the platform
    """
    ids_number_test = [i.split("ID_")[1] for i in ids_test]
    test_output = pd.DataFrame({"ID": ids_number_test, "Target": predictions})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv(data_dir / "preds_test_baseline.csv")


def get_params(args):
    """
    Piece of code taken from the baselines for usage in the trainer file
    Get the train and test set + labels + test ids for the submission files
    """
    # -------------------------------------------------------------------------
    # Load the data
    assert args.data_dir.is_dir()

    train_dir = args.data_dir / "train_input" / "resnet_features"
    test_dir = args.data_dir / "test_input" / "resnet_features"

    train_output_filename = args.data_dir / "train_output.csv"

    train_output = pd.read_csv(train_output_filename)

    # Get the filenames for train
    filenames_train = [train_dir / "{}.npy".format(idx) for idx in train_output["ID"]]
    for filename in filenames_train:
        assert filename.is_file(), filename

    # Get the labels
    labels_train = train_output["Target"].values

    assert len(filenames_train) == len(labels_train)

    # Get the numpy filenames for test
    filenames_test = sorted(test_dir.glob("*.npy"))
    for filename in filenames_test:
        assert filename.is_file(), filename
    ids_test = [f.stem for f in filenames_test]
    return filenames_train, filenames_test, labels_train, ids_test


def get_model(mtype, **kwargs):
    """
    Create the model depending on the current params
    """
    if mtype == 'normal':
        model = EnsembleModel(**kwargs)
    elif mtype == 'multi_r':
        model_list = [ChowderModel(r) for r in kwargs['R_list']]
        model = EnsembleModel.from_model_list(model_list)
    else:
        raise NotImplementedError
    return model


def get_optimizer(model, train_together=baseCONFIG.train_together):
    """
    Build the pytorch optimizer based on the current params
    """
    if train_together:
        optimizer = optim.Adam(sum([[{'params': m.projector.parameters(), 'weight_decay': baseCONFIG.l2_penalty},
                                     {'params': m.mlp.parameters()}] for m in model.model_list], []),
                               lr=baseCONFIG.lr)
    else:
        optimizer = [optim.Adam([{'params': m.projector.parameters(), 'weight_decay': baseCONFIG.l2_penalty},
                                 {'params': m.mlp.parameters()}
                                 ],
                                lr=baseCONFIG.lr) for m in model.model_list]
    return optimizer


def get_model_and_optimizer(mtype, model_params):
    """
    Create model and optimizer
    :param mtype: model type : normal or multi_r
    :param cuda: boolean, use GPU
    :param l2_penalty: float, weight og the l2_penalty (0: None)
    :param kwargs: Parameter for EnsembleModel
    :return:
    """
    model = get_model(mtype=mtype, **model_params)
    if baseCONFIG.cuda:
        model.cuda()
    optimizer = get_optimizer(model, train_together=baseCONFIG.train_together)
    return model, optimizer


def train_on_one_epoch(**kwargs):
    """
    Wrapper function that train the model for one epoch
    """
    if baseCONFIG.train_together:
        train_full_model_on_one_epoch(**kwargs)
    else:
        train_each_model_on_one_epoch(**kwargs)


def train_each_model_on_one_epoch(model, train_dataset, optimizer, dataloader=custom_dataloader, verbose=True):
    """
    Train the ensemble model on one epoch by training each submodel separately.
    Different batches are used for each model
    :param model: EnsembleModel
    :param train_dataset: dataset containing the training samples and their labels
    :param optimizer: list of optimizer, one for each submodel
    :param batch_size: int
    :param dataloader: pytorch dataloader to be used
    :param verbose: boolean, print mean loss on epoch
    :param cuda: boolean, use gpu for computation
    :return: nothing, print the mean loss for each submodel
    """
    loss_fn = nn.BCELoss()
    try:
        train_loader = dataloader(dataset=train_dataset, batch_size=baseCONFIG.batch_size, shuffle=True, num_workers=4)
    except ValueError:
        train_loader = dataloader(dataset=train_dataset, batch_size=baseCONFIG.batch_size, num_workers=4)
    for model_, optim_ in zip(model.model_list, optimizer):
        optim_.zero_grad()
        losses = []
        for _, sample_batched in enumerate(train_loader):
            features, labels = sample_batched
            if baseCONFIG.cuda:
                features, labels = features.cuda(), labels.cuda()
            predictions = model_(features)
            loss = loss_fn(predictions, labels.float())
            loss.backward()
            optim_.step()
            losses.append(loss.detach())
        if verbose:
            print(f'mean loss on epoch : {torch.mean(torch.tensor(losses).cpu())}')


def train_full_model_on_one_epoch(model, train_dataset, optimizer, dataloader=custom_dataloader, verbose=True):
    """
    Train the ensemble model on one epoch by training the ensemble model at once.
    :param model: EnsembleModel
    :param train_dataset: dataset containing the training samples and their labels
    :param optimizer: pytorch optimizer
    :param batch_size: int
    :param dataloader: pytorch dataloader to be used
    :param verbose: boolean, print mean loss on epoch
    :param cuda: boolean, use gpu for computation
    :return: nothing, print the mean loss for each submodel
    """
    loss_fn = nn.BCELoss()
    try:
        train_loader = dataloader(dataset=train_dataset, batch_size=baseCONFIG.batch_size, shuffle=True, num_workers=4)
    except ValueError:
        train_loader = dataloader(dataset=train_dataset, batch_size=baseCONFIG.batch_size, num_workers=4)
    optimizer.zero_grad()
    losses = []
    for _, sample_batched in enumerate(train_loader):
        features, labels = sample_batched
        if baseCONFIG.cuda:
            features, labels = features.cuda(), labels.cuda()
        predictions = model.predict(features)
        loss = loss_fn(predictions, labels.float())
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    if verbose:
        print(f'mean loss on epoch : {torch.mean(torch.tensor(losses).cpu())}')


def eval_model(model, val_dataset, name='validation', dataloader=custom_dataloader, verbose=True):
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
        if baseCONFIG.cuda:
            val_features = val_features.cuda()
        val_pred = model.predict(val_features).detach()
        predictions.append(val_pred)
    predictions, labels = torch.cat(predictions).cpu().numpy(), torch.cat(labels).cpu().numpy()
    auc_score = roc_auc_score(y_true=labels, y_score=predictions)
    if verbose:
        print(f'{name} AUC: {auc_score}')
    model.train()
    return auc_score


def eval_on_test(model, dataset):
    """
    Compute the predictions of the model on dataset
    """
    test_loader = custom_dataloader(dataset=dataset, batch_size=10, shuffle=False, num_workers=4)
    predictions = []
    for _, (features, _) in enumerate(test_loader):
        if baseCONFIG.cuda:
            features = features.cuda()
        predictions.append(model.predict(features).detach())
    predictions = torch.cat(predictions).cpu().numpy()
    return predictions
