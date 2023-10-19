import os, json, pickle
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from argparse import ArgumentParser
from schemes import iid_partition, dirichlet, pathological
from datasets import DatasetDict, CustomSubset
from setting import par_dict, data_dict

def partition_data(dataset, args):
    '''
    niid: non-iid, the classes for client are random (while the number of samples is fixed [this can be appointed]), include 'balance', which is about the number
    balance: if True, the number of samples for each class is the same
    partition: two options['pat': pathologic, 'dir': Dirichlet, 'mix': mix]
    1. if niid, partion is useless. 2. if partition is dir, balance is useless.
    '''
    labels = dataset.targets
    num_client, iid, balance, partition, alpha, num_class_client, least_samples = args.num_client, args.iid, args.balance, args.partition, args.alpha, args.num_class_client, args.least_samples
    
    if iid:
        idx_sample_client = iid_partition(dataset, num_client, num_class_client, balance, least_samples)
    
    elif partition == 'pat':
        idx_sample_client = pathological(dataset, num_client, num_class_client, balance, least_samples)
        
    elif partition == 'dir':
        idx_sample_client = dirichlet(dataset, num_client, alpha)
    else:
        raise NotImplementedError
    
    # lazyness, global statistic, no train/test split
    statistic_client = [[] for _ in range(num_client)]
    for i in range(num_client):
        label_client = labels[idx_sample_client[i]]
        for label in np.unique(label_client):
            statistic_client[i].append((int(label), int(sum(label_client==label))))

    idx_sample_client = split_data(idx_sample_client, train_size=args.train_size)
    
    return idx_sample_client, statistic_client

def split_data(partition, train_size=0.8):
    for client, idx_sample_client in enumerate(partition):
        num_train = int(len(idx_sample_client) * train_size)
        np.random.shuffle(idx_sample_client)
        idx_train, idx_test = idx_sample_client[:num_train], idx_sample_client[num_train:]
        partition[client] = {'train': idx_train, 'test': idx_test}

        # num_train, num_test = len(partition[client]['train']), len(partition[client]['test'])
        # print("Total number of samples:", num_train + num_test)
        # print("The number of train samples:", num_train)
        # print("The number of test samples:", num_test)
    return partition

def save_partition(partition, stats, args, path):
    with open(os.path.join(path, "partition.pkl"), "wb") as f:
        pickle.dump(partition, f)

    with open(os.path.join(path, "stats.json"), "w") as f:
        json.dump(stats, f)

    with open(os.path.join(path, "args.json"), "w") as f:
        json.dump(vars(args), f)
    
def draw_distribution(dataset,  partition):
    labels = dataset.targets
    name_class = dataset.classes
    num_client = len(partition)
    num_class = len(name_class)

    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(num_class)]
    for client, idx_sample in enumerate(partition):
        for idx in idx_sample:
            label_distribution[labels[idx]].append(client)

    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, num_client + 1.5, 1),
             label=name_class, rwidth=0.5)
    plt.xticks(np.arange(num_client), ["Client %d" % c_id for c_id in range(num_client)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend(loc="upper right")
    plt.title("Display Label Distribution on Different Clients")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['minst'])
    parser.add_argument('-n', '--num_client', type=int, default=10)
    parser.add_argument('--iid', type=int, default=0)
    parser.add_argument('--balance', type=int, default=1)
    parser.add_argument('--partition', type=str, choices=['pat', 'dir', 'mix'], default='pat')
    parser.add_argument('-a', '--alpha', type=float, default=0.1)
    parser.add_argument('-nc', '--num_class_client', type=int, default=3)
    parser.add_argument('-ls', '--least_samples', type=int, default=40)
    parser.add_argument('-ts', '--train_size', type=float, default=0.8)
    args = parser.parse_args()

    dataset_name = 'mnist'
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    dataset = DatasetDict[dataset_name](root=data_dict[dataset_name], transform=transform)

    partition, stats = partition_data(dataset, args)
    save_partition(partition, stats, args, path=par_dict[dataset_name])
    # draw_distribution(dataset, partition)
    with open(os.path.join(par_dict[dataset_name], 'partition.pkl'), "rb") as f:
        partition = pickle.load(f)
    trainset = CustomSubset(dataset, partition[0]['train'])
    print(trainset.data.shape)