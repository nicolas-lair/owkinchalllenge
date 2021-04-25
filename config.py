from pathlib import Path
import torch


class baseCONFIG:
    """
    Shared config params accross simulations
    """
    data_dir = Path().absolute()

    # Train params
    batch_size = 10
    epoch = 30
    lr = 0.001
    train_together = True

    # Regularization params
    batchnorm = True
    regularization = 'light'
    if regularization == 'no':
        dropout = [0, 0, 0]
        l2_penalty = 0
    elif regularization == 'light':
        dropout = [0, 0.25, 0.25]
        l2_penalty = 0.3
    elif regularization == 'normal':
        dropout = [0, 0.5, 0.5]
        l2_penalty = 0.5
    else:
        raise NotImplementedError

    # Cuda
    cuda = torch.cuda.is_available()
    if cuda:
        device = 'cuda:2'

    # Cross val params
    num_runs = 3
    num_splits = 5

    def __repr__(self):
        import inspect
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        print([a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))])


class Chowder_CONFIG(baseCONFIG):
    """
    Config for Chowder model as presented in the paper
    """
    mtype = 'normal'
    model_params = dict(E=10, R=5)


class multiR_CONFIG(baseCONFIG):
    """
    Config for an ensemble Chowder model with different R value
    """
    mtype = 'multi_r'
    model_params = dict(R_list=[1, 3, 5, 7, 10, 20, 30, 50, 75, 100])


class DeepSet_CONFIG(baseCONFIG):
    mtype = 'deepset'
    model_params = dict(scaler_size=256)
