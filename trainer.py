from torch.utils.data import random_split

from utils import get_params, save_test_predictions, get_model_and_optimizer, train_on_one_epoch, eval_model, \
    eval_on_test
from loader import ResNetFeaturesDataset
from config import Chowder_CONFIG, multiR_CONFIG

config = Chowder_CONFIG
# config = multiR_CONFIG
compute_test = True

if __name__ == '__main__':
    # Prepare the simulation
    filenames_train, filenames_test, labels_train, ids_test = get_params(config)
    TrainDataset = ResNetFeaturesDataset(filenames=filenames_train, labels=labels_train)
    p = 0.91
    train_dataset, val_dataset = random_split(TrainDataset, lengths=[int(p * len(TrainDataset)),
                                                                     len(TrainDataset) - int(
                                                                         p * len(TrainDataset))])

    # Create the models
    model, optimizer = get_model_and_optimizer(mtype=config.mtype, model_params=config.model_params)

    # Train
    train_auc = []
    val_auc = []
    predictions = dict()
    for e in range(config.epoch):
        print('-' * 5 + f' Epoch {e} ' + '-' * 5)
        print('-- Training')
        train_on_one_epoch(model=model, train_dataset=train_dataset, optimizer=optimizer)
        print(' -- Evaluation')
        train_auc.append(eval_model(model, train_dataset, name='train set'))
        val_auc.append(eval_model(model, val_dataset, name='validation set'))

        if compute_test:
            # Compute predictions on the test set
            print('-- Computing the predictions on the test set')
            TestDataset = ResNetFeaturesDataset(filenames=filenames_test)
            predictions[e] = eval_on_test(model=model, dataset=TestDataset)

    # if compute_test:
    #     print(f'Predictions on the test were computed for AUC on val of {max_val_auc}')
    #     save_test_predictions(ids_test=ids_test, predictions=predictions, data_dir=config.data_dir)
