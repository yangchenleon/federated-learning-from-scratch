import os, json
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split

def partition_data(dataset, num_client, niid=True, balance=False, partition=None, alpha=0.1, num_class_client=2, least_samples=100):
    '''
    niid: non-iid, the classes for client are random (while the number of samples is fixed [this can be appointed]), include 'balance', which is about the number
    balance: if True, the number of samples for each class is the same
    partition: two options['pat': pathologic, 'dir': Dirichlet, 'mix': mix]
    1. if niid, partion is useless. 2. if partition is dir, balance is useless.
    '''
    _, labels = dataset
    del dataset
    num_class = len(np.unique(labels))
    idx_sample_client = [[] for _ in range(num_client)]

    if not niid:
        partition = 'pat'
        num_class_client = num_class
    
    if partition == 'pat':
        # class data by class
        idxs = np.array(range(len(labels)))
        label2idx = [None for _ in range(num_class)]
        for i in range(num_class):
            label2idx[i] = idxs[labels==i]
        
        idx_client = np.array(range(num_client))
        cnt_class_client = np.array([num_class_client for _ in range(num_client)])
        available_client = np.array(range(num_client))

        for i in range(num_class):
            # choose client for class i and update client list
            num_select = int(np.ceil((num_client/num_class)*num_class_client))
            # All the data is divided into num_client * class_per_client shards, with each class evenly distributed among these shards (dividing them by num_class). The number of shards in class corresponds to the number of clients selcected.
            pri_selected_client = idx_client[cnt_class_client==max(cnt_class_client)]
            if pri_selected_client.size >= num_select:
                selected_client = np.random.choice(pri_selected_client, num_select, replace=False)
            else:
                restavai_client = np.setdiff1d(available_client, pri_selected_client)
                aft_selected_client = np.random.choice(restavai_client, num_select-len(pri_selected_client), replace=False)
                selected_client = np.concatenate((pri_selected_client, aft_selected_client))
            cnt_class_client[selected_client] -= 1
            available_client = idx_client[cnt_class_client>0]

            # now we have selected clients and data(sample) of class i
            idx_sample = label2idx[i]
            num_sample, num_slected_client = len(idx_sample), len(selected_client)
            num_sample_client_avg = int(num_sample/num_slected_client)

            # choose samples for each client, whether balance or not
            if balance:
                num_sample_client = np.ones(num_slected_client, dtype=int) * num_sample_client_avg
                num_sample_client[:(num_sample - sum(num_sample_client))] += 1
            else:
                least_sample_client = least_samples / num_slected_client
                num_sample_client = np.random.randint(max(num_sample_client_avg/10, least_sample_client), num_sample_client_avg, num_slected_client-1)
                num_sample_client = np.append(num_sample_client, num_sample - sum(num_sample_client[:-1]))
            # assign sample to client with idx
            idx = 0
            for client, num_sample_client in zip(selected_client, num_sample_client):
                idx_sample_client[client].extend(idx_sample[idx:idx+num_sample_client])
                idx += num_sample_client
    elif partition == 'dir':
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_class
        N = len(labels)

        while min_size < num_class:
            idx_batch = [[] for _ in range(num_client)]
            for k in range(K):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_client))
                proportions = np.array([p*(len(idx_j)<N/num_client) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_client):
            idx_sample_client[j] = idx_batch[j]
    else:
        raise NotImplementedError
    # del dataset
    return idx_sample_client

def assign_data(dataset, partition):
    inputs, labels = dataset
    num_client = len(partition)
    input_client = [[] for _ in range(num_client)]
    label_client = [[] for _ in range(num_client)]
    statistic_client = [[] for _ in range(num_client)]
    for i in range(num_client):
        input_client[i] = inputs[partition[i]]
        label_client[i] = labels[partition[i]]
        # statistic_client[i] = np.unique(label_client[i], return_counts=True)
        for label in np.unique(label_client[i]):
            statistic_client[i].append((int(label), int(sum(label_client[i]==label))))
    del dataset

    for i in range(num_client):
        print(f"Client {i}\t Size of data: {len(input_client[i])}\t Labels: ", np.unique(label_client[i]))
        print(f"\t\t Samples of labels: ", [i for i in statistic_client[i]])
        print("-" * 80)

    return input_client, label_client, statistic_client

def split_data(dataset, train_size):
    '''
    per client, not from system aspect
    '''
    input, label = dataset
    X_train, X_test, y_train, y_test = train_test_split(input, label, train_size=train_size, shuffle=True)

    # num_train, num_test = len(y_train), len(y_test)
    # print("Total number of samples:", num_train + num_test)
    # print("The number of train samples:", num_train)
    # print("The number of test samples:", num_test)

    return {'x': X_train, 'y': y_train}, {'x': X_test, 'y': y_test}

def save_partition(dataset, partition, param, path):
    num_client = len(partition)
    input_client, label_client, statistic_client = assign_data(dataset, partition)
    
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(os.path.join(path, 'train')):
        os.makedirs(os.path.join(path, 'train'))
    if not os.path.exists(os.path.join(path, 'test')):
        os.makedirs(os.path.join(path, 'test'))
    
    for i in range(num_client):
        train_data, test_data = split_data((input_client[i], label_client[i]), param['train_size'])
        np.savez_compressed(os.path.join(path, f'train/{i}.npz'), data=train_data)
        np.savez_compressed(os.path.join(path, f'test/{i}.npz'), data=test_data)
    
    config = param.copy()
    config.update({'num_client': num_client, 'statistic_client': statistic_client})
    with open(os.path.join(path, f'config.json'), 'w') as f:
        json.dump(config, f)
    return 
    
def draw_distribution(dataset, name_class, partition):
    _, labels = dataset
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
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    train_data = torchvision.datasets.MNIST(
        root="datasets", download=True, train=True, transform=transform)
    test_data = torchvision.datasets.MNIST(
        root="datasets", download=True, train=False, transform=transform)

    name_class = train_data.classes
    
    dataset = torch.utils.data.ConcatDataset([train_data, test_data])
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    dataset_image, dataset_label = next(iter(dataset_loader))
    dataset_image, dataset_label = np.array(dataset_image), np.array(dataset_label)
    
    np.random.seed(42)
    num_client = 10
    partition_setting = json.loads(open('datasets/setting.json').read())
    partition = partition_data((dataset_image, dataset_label), num_client, **partition_setting)
    # draw_distribution((dataset_image, dataset_label), name_class, partition)
    save_partition((dataset_image, dataset_label), partition, partition_setting, path='datasets/_ClientData')

# -----------------
# 以上是首次尝试，有点杂，大部分参考Non-IID-PFL，pat划分不太对的应该是，dir直接copy，很多参数都没用上，但是放在同一个函数下，后面就分开了；以及直接保存的划分数据集，保存和读取起来有点笨重，后面仅保存partition pkl，即用即读。
# 后面是再修改，已经拆分模块了，但是感觉还是有问题，包括iid的划分直接通用pat的方法，而pat方法本身就感觉有点不靠谱，dir方法一方面算法本身没有利用一些参数，虽然说是官方的，总感觉还可以改进。
# --------- 
def iid_partition(dataset, num_client, balance, least_samples):
    labels = dataset.targets
    num_class = len(dataset.classes)
    idx_sample_client = [[] for _ in range(num_client)]

    idx_sample_client = pathological(dataset, num_client, num_class_client=num_class, balance=balance, least_samples=least_samples)

    return idx_sample_client

# -------------
# global statistic, no train/test split, also actually no need, cause the train/test in a client usually is iid
statistic_client = [[] for _ in range(args.num_client)]
for i in range(args.num_client): # [client1[(label1, num_label1),()...], client2[()]] 
    label_client = labels[idx_sample_client[i]]
    for label in np.unique(label_client):
        statistic_client[i].append((int(label), int(sum(label_client==label))))