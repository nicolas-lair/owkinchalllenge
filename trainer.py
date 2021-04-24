from torch.utils.data import random_split

from utils import get_params, save_test_predictions, get_model_and_optimizers, train_on_one_epoch, eval_model, \
    eval_on_test
from loader import ResNetFeaturesDataset
from config import Chowder_CONFIG, multiR_CONFIG

# config = Chowder_CONFIG
config = multiR_CONFIG
compute_test = False

if __name__ == '__main__':
    # Prepare the simulation
    filenames_train, filenames_test, labels_train, ids_test = get_params(config)
    TrainDataset = ResNetFeaturesDataset(filenames=filenames_train, labels=labels_train)
    p = 0.8
    train_dataset, val_dataset = random_split(TrainDataset, lengths=[int(p * len(TrainDataset)),
                                                                     len(TrainDataset) - int(
                                                                         p * len(TrainDataset))])

    # Create the models
    model, optimizers = get_model_and_optimizers(mtype=config.mtype, model_params=config.model_params)

    # Train
    max_val_auc = 0
    for e in range(config.epoch):
        print('-' * 5 + f' Epoch {e} ' + '-' * 5)
        print('-- Training')
        train_on_one_epoch(model, train_dataset, optimizers)
        print(' -- Evaluation')
        eval_model(model, train_dataset, name='train set')
        val_auc = eval_model(model, val_dataset, name='validation set')

        if compute_test and val_auc > max_val_auc:
            # Compute predictions on the test set
            print('-- Computing the predictions on the test set')
            TestDataset = ResNetFeaturesDataset(filenames=filenames_test)
            predictions = eval_on_test(model=model, dataset=TestDataset)
            max_val_auc = val_auc

    if compute_test:
        print(f'Predictions on the test were computed for AUC on val of {max_val_auc}')
        save_test_predictions(ids_test=ids_test, predictions=predictions, data_dir=config.data_dir)
