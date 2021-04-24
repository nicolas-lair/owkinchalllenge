from pathlib import Path
import torch


class baseCONFIG:
    data_dir = Path().absolute()

    # Train params
    batch_size = 10
    epoch = 30
    lr = 0.001
    train_together = True

    # Regularization params
    dropout = [0, 0.25, 0.2]
    l2_penalty = 0.3

    # Cuda
    cuda = torch.cuda.is_available()

    # Cross val params
    num_runs = 3
    num_splits = 5


class Chowder_CONFIG(baseCONFIG):
    mtype = 'normal'
    model_params = dict(E=10, R=5)


class multiR_CONFIG(baseCONFIG):
    mtype = 'multi_r'
    model_params = dict(R_list=[1, 3, 5, 7, 10, 20, 30, 50, 75, 100])
