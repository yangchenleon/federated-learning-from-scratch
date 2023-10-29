import numpy as np
from argparse import ArgumentParser

from data.utils.dataset_utils import partition_data
from src.server.fedavg import FedAvgServer


if __name__ == "__main__":
    np.random.seed(42)
    parser = ArgumentParser()
    parser.add_argument('-n', '--num_client', type=int, default=10)
    parser.add_argument('--balance', type=int, default=1) # actually only impl pat imbalance
    parser.add_argument('--partition', type=str, choices=['iid', 'pat', 'dir', 'mix', 'rad', 'srd'], default='iid')
    parser.add_argument('-a', '--alpha', type=float, default=0.1)
    parser.add_argument('-nc', '--num_class_client', type=int, default=7)
    parser.add_argument('-ls', '--least_samples', type=int, default=40)
    parser.add_argument('-ts', '--train_size', type=float, default=0.8)

    # mlp: 0.1, cnn: 0.001, alexnet/resnet: 0.01, 
    parser.add_argument('-pt', '--pretrained', type=int, default=0)
    parser.add_argument('-lr', '--lr', type=float,  default=1e-2)
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-mom', '--momentum', type=float, default=0.9)
    parser.add_argument('-ne', '--num_epochs', type=int, default=5)

    parser.add_argument('-jt', '--join_ratio', type=float, default=0.2)
    parser.add_argument('-gr', '--global_round', type=int, default=25)

    args = parser.parse_args()

    datasets = ['cifar10']
    models = ['theres18']
    partition, stats = partition_data(datasets[0], args, draw=True)
    # exit()
  
    server = FedAvgServer(datasets, models, args)
    server.train()