import os, json, pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from data.utils.setting import par_dict
from data.utils.datasets import DatasetDict
from data.utils.schemes import *


def partition_data(dataset, args, save=True, draw=False):
    '''
    niid: non-iid, the classes for client are random (while the number of samples is fixed [this can be appointed]), include 'balance', which is about the number
    balance: if True, the number of samples for each class is the same
    partition: two options['pat': pathologic, 'dir': Dirichlet, 'mix': mix]
    1. if niid, partion is useless. 2. if partition is dir, balance is useless.
    '''
    path = par_dict[dataset]
    if check(args, path):
        with open(os.path.join(path, "partition.pkl"), "rb") as f:
            split_partition =  pickle.load(f)
        with open(os.path.join(path, "stats.json"), "r") as f:
            statistic_client = json.load(f)
        return split_partition, statistic_client
    
    dataset = DatasetDict[dataset](transform=None)
    if args.partition == 'iid':
        idx_sample_client = iid_partition(dataset, args.num_client)
    elif args.partition == 'pat':
        idx_sample_client = pathological(dataset, args.num_client, args.num_class_client, args.balance, args.least_samples)
    elif args.partition == 'dir':
        idx_sample_client = dirichlet(dataset, args.num_client, args.alpha, args.least_samples)
    elif args.partition == 'rad':
        idx_sample_client = randomly_assign_classes(dataset, args.num_client, args.num_class_client)
    elif args.partition == 'srd':
        idx_sample_client = allocate_shards(dataset, args.num_client, shard_num=args.num_class_client)
    else:
        raise NotImplementedError
    
    partition = idx_sample_client
    statistic_client = statistic(dataset, partition, args)
    split_partition = split_data(idx_sample_client, train_size=args.train_size)
        
    if save:
        save_partition(split_partition, statistic_client, args, path)
    
    return split_partition, statistic_client

def split_data(partition, train_size=0.8):
    split_partition = [None for _ in range(len(partition))]
    for client, idx_sample_client in enumerate(partition):
        num_train = int(len(idx_sample_client) * train_size)
        np.random.shuffle(idx_sample_client)
        idx_train, idx_test = idx_sample_client[:num_train], idx_sample_client[num_train:]
        split_partition[client] = {'train': idx_train, 'test': idx_test}
        # amazing here, if inplace operation (partition[client]={'train'...}), though new_place = split_data，the input para partition will change

    return split_partition

def statistic(dataset, partition, args):
    stats = {}
    labels = dataset.targets

    for i in range(args.num_client): # [client1[(label1, num_label1),()...], client2[()]]
        label_client = labels[partition[i]]
        stats[i] = {}
        stats[i]['num_sample'] = len(label_client)
        stats[i]['class_count'] = Counter(label_client.tolist())
    num_samples = np.array(list(map(lambda stat_i: stat_i['num_sample'], stats.values())))
    stats["sample_per_client"] = {"std": num_samples.mean(), "stddev": num_samples.std()}
    # honestly dont know what sample_per_client for 

    return stats

def save_partition(partition, stats, args, path):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, "partition.pkl"), "wb") as f:
        pickle.dump(partition, f)

    with open(os.path.join(path, "stats.json"), "w") as f:
        json.dump(stats, f)

    with open(os.path.join(path, "args.json"), "w") as f:
        json.dump(vars(args), f)

def check(args, path):
    # false mean not exist
    if not os.path.exists(os.path.join(path, "partition.pkl")) \
        or not os.path.exists(os.path.join(path, "args.json")):
        return False

    keys = ['num_client', 'balance', 'partition', 'alpha', 'num_class_client', 'least_samples', 'train_size']
    with open(os.path.join(path, "args.json"), "r") as f:
        data = json.load(f)
        for key in keys:
            if data[key] != getattr(args, key):
                return  False
    return True