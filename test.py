import os, pickle
import numpy as np
import torch, torchvision
from argparse import ArgumentParser
from torchvision import transforms
from data.utils.datasets import DatasetDict, CustomSubset
from data.utils.setting import par_dict, data_dict
from data.utils.dataset_utils import partition_data, save_partition, draw_distribution
from src.client.fedavg import Client

if __name__ == "__main__":
    np.random.seed(42)
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['minst'])
    parser.add_argument('-n', '--num_client', type=int, default=1)
    parser.add_argument('--iid', type=int, default=0)
    parser.add_argument('--balance', type=int, default=1)
    parser.add_argument('--partition', type=str, choices=['pat', 'dir', 'mix'], default='pat')
    parser.add_argument('-a', '--alpha', type=float, default=0.1)
    parser.add_argument('-nc', '--num_class_client', type=int, default=3)
    parser.add_argument('-ls', '--least_samples', type=int, default=40)
    parser.add_argument('-ts', '--train_size', type=float, default=0.8)

    # mlp: 0.1, alexnet: 0.01
    parser.add_argument('-lr', '--lr', type=float,  default=1e-2)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-bs", "--batch_size", type=int, default=256)
    parser.add_argument("-mom", "--momentum", type=float, default=0.0)
    parser.add_argument("-ne", "--num_epochs", type=int, default=10)
    args = parser.parse_args()

    dataset_name, model_name = 'mnist', 'pretrianed-alexnet'
    # partition, stats = partition_data(dataset_name, args)
    # save_partition(partition, stats, args, path=par_dict[dataset_name])
    # draw_distribution(dataset, partition)
    # with open(os.path.join(par_dict[dataset_name], 'partition.pkl'), "rb") as f:
    #     partition = pickle.load(f)
    
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, antialias=True),
        transforms.Normalize([0.5], [0.5]),
        # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    client = Client(0, dataset_name, model_name, args, None, device)
    trainset, testset = client.load_dataset(transform=trans)
    client.trainloader = torch.utils.data.DataLoader(CustomSubset(trainset, range(100)), batch_size=16, shuffle=False) if torch.cuda.is_available() else client.trainloader
    client.train()
    # client.load_state('results/checkpoints/0_mnist_AlexNet_40.pth')
    client.eval()

    # 更新一下partition的读取问题
    # 训练过程、预测结果可视化
    # index相关问题（数据索引是否正确）
    # pretrained loading
    # 中间特征输出