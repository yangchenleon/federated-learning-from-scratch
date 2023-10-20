import os, pickle
import numpy as np
import torch, torchvision
from argparse import ArgumentParser
from torchvision import transforms
from data.utils.datasets import DatasetDict, CustomSubset
from data.utils.setting import par_dict, data_dict
from data.utils.dataset_utils import partition_data, save_partition, draw_distribution
from src.client.fedavg import Client

from models.models import ModelDict
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader

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

    parser.add_argument('-lr', '--lr', type=float,  default=1e-1)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-mom", "--momentum", type=float, default=0.0)
    args = parser.parse_args()

    dataset_name = 'fashion'
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    dataset = DatasetDict[dataset_name](root=data_dict[dataset_name], transform=transform)
    # partition, stats = partition_data(dataset, args)
    # save_partition(partition, stats, args, path=par_dict[dataset_name])
    # draw_distribution(dataset, partition)
    with open(os.path.join(par_dict[dataset_name], 'partition.pkl'), "rb") as f:
        partition = pickle.load(f)
    trainset = CustomSubset(dataset, partition[0]['train'])
    testset = CustomSubset(dataset, partition[0]['test'])
    
    # dataset_name, model_name = 'mnist', 'mlp'
    # client = Client(0, dataset_name, model_name, args, None, 'cpu')
    # trainset, testset = client.load_dataset(transform=None)
    # client.train(10)
    # client.eval()

    # train the models
    train_iter = DataLoader(trainset, batch_size=256, shuffle=True)
    test_iter = DataLoader(testset, batch_size=256, shuffle=False)

    net = ModelDict['mlp'](dataset_name)
    lr, num_epochs = 0.1, 10
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('training on', device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    criteria = nn.CrossEntropyLoss()


    for epoch in range(num_epochs):
        
        net.train()
        ls = 0
        for X, y in train_iter:
            # print(y)
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            loss = criteria(net(X), y)
            loss.backward()
            optimizer.step()
            ls += loss.item()
        if epoch % 1 == 0:
            print('epoch', epoch)
            print(ls/len(y))
    net.eval()
    X, y = next(iter(test_iter))
    with torch.no_grad():
        preds = net(X).argmax(axis=1)
    
    print(f'{sum(preds==y)/len(y):2f}')