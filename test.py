import numpy as np
from torchvision import transforms
from argparse import ArgumentParser

from data.utils.dataset_utils import partition_data
from src.server.fedavg import FedAvgServer


if __name__ == "__main__":
    np.random.seed(42)
    parser = ArgumentParser()
    parser.add_argument('-n', '--num_client', type=int, default=20)
    parser.add_argument('--balance', type=int, default=1) # actually only impl pat imbalance
    parser.add_argument('--partition', type=str, choices=['iid', 'pat', 'dir', 'mix', 'rad', 'srd'], default='iid')
    parser.add_argument('-a', '--alpha', type=float, default=0.1)
    parser.add_argument('-nc', '--num_class_client', type=int, default=7)
    parser.add_argument('-ls', '--least_samples', type=int, default=40)
    parser.add_argument('-ts', '--train_size', type=float, default=0.8)

    # mlp: 0.1, alexnet: 0.01, cnn: 0.001
    parser.add_argument('-pt', '--pretrained', type=int, default=0)
    parser.add_argument('-lr', '--lr', type=float,  default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-mom', '--momentum', type=float, default=0.0)
    parser.add_argument('-ne', '--num_epochs', type=int, default=5)

    parser.add_argument('-jt', '--join_ratio', type=float, default=0.1)
    parser.add_argument('-gr', '--global_round', type=int, default=2)

    args = parser.parse_args()

    datasets = ['mnist']
    models = ['cnn']
    partition, stats = partition_data(datasets[0], args, draw=True)
    # exit()

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616]),
        # transforms.Resize(224, antialias=True),
    ])
    
    server = FedAvgServer(datasets, models, args)
    server.load_testset(transform=trans)
    server.train()


    # log，即各种参数设置
    # 中间特征输出
    # 有点偏：训练一半minst跑去训练cifar10，很怪但是想实现这个log —— optional
    # set default root for dataset
    # update server maintained clients' trained epoch