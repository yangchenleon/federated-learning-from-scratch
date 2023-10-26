import os, json, pickle
import numpy as np
import matplotlib.pyplot as plt

from data.utils.datasets import DatasetDict
from data.utils.schemes import *
from data.utils.setting import par_dict

def partition_data(dataset, args, save=True, draw=False):
    '''
    niid: non-iid, the classes for client are random (while the number of samples is fixed [this can be appointed]), include 'balance', which is about the number
    balance: if True, the number of samples for each class is the same
    partition: two options['pat': pathologic, 'dir': Dirichlet, 'mix': mix]
    1. if niid, partion is useless. 2. if partition is dir, balance is useless.
    '''
    path = par_dict[dataset]
    dataset = DatasetDict[dataset](transform=None)
    labels = dataset.targets
    
    if args.partition == 'iid':
        idx_sample_client = iid_partition(dataset, args.num_client)
    elif args.partition == 'pat':
        idx_sample_client = pathological(dataset, args.num_client, args.num_class_client, args.balance, args.least_samples)
    elif args.partition == 'dir':
        idx_sample_client = dirichlet(dataset, args.num_client, args.alpha)
    elif args.partition == 'rad':
        idx_sample_client = randomly_assign_classes(dataset, args.num_client, args.num_class_client)
    elif args.partition == 'srd':
        idx_sample_client = allocate_shards(dataset, args.num_client, shard_num=args.num_class_client)
    else:
        raise NotImplementedError
    
    # global statistic, no train/test split, also actually no need, cause the train/test in a client usually is iid
    statistic_client = [[] for _ in range(args.num_client)]
    for i in range(args.num_client): # [client1[(label1, num_label1),()...], client2[()]] 
        label_client = labels[idx_sample_client[i]]
        for label in np.unique(label_client):
            statistic_client[i].append((int(label), int(sum(label_client==label))))

    idx_sample_client = split_data(idx_sample_client, train_size=args.train_size)

    if save:
        save_partition(idx_sample_client, statistic_client, args, path)
    if draw:
        draw_distribution(dataset, idx_sample_client, args, path)
    
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
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, "partition.pkl"), "wb") as f:
        pickle.dump(partition, f)

    with open(os.path.join(path, "stats.json"), "w") as f:
        json.dump(stats, f)

    with open(os.path.join(path, "args.json"), "w") as f:
        json.dump(vars(args), f)
    
def draw_distribution(dataset, partition, args, path):
    labels = dataset.targets
    name_class = dataset.classes
    num_client = len(partition)
    num_class = len(name_class)
    partition = [partition[i]['train'] for i in range(num_client)] #  + partition[i]['test'], but one is enough

    if not os.path.exists(path):
        os.makedirs(path)

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
    plt.savefig(os.path.join(path, f'{type(dataset).__name__[6:]}_{args.num_client}_{args.partition}.png'))
    plt.show()

