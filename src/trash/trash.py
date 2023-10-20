'''
gen_mnist.py
'''
import os, json
import numpy as np
import torch, torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

# four file: t10k for test, train for train with labels and images
trainset = getattr(torchvision.datasets, 'MNIST')(
    root='./datasets', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(
    root='./datasets', train=False, download=True, transform=transform)
# v1：本来想用这种方法的，但是这种循环的方法应该不好, 
# 同时concatDataset直接喂给dataloader的话应该也可以
dataset = torch.utils.data.ConcatDataset([trainset, testset])
dataset_image = torch.stack([sample[0] for sample in dataset])
dataset_label = torch.tensor([sample[1] for sample in dataset])

# v2：原版用dataloader读取的，在第二维增加了一个channel维，这里手动增加，感觉没必要通过dataloader+enumerata的方式实现
dataset_image = torch.cat((torch.unsqueeze(trainset.data, dim=1), 
                           torch.unsqueeze(testset.data, dim=1)), dim=0)
dataset_label = torch.cat((trainset.targets, testset.targets), dim=0)
X_train, X_test, y_train, y_test = train_test_split(dataset_image, dataset_label, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)

# dataset_image = np.array(dataset_image)
# dataset_label = np.array(dataset_label)

# v3：dataloader对多种数据集融合效果更好（可以是其他数据集，这里融合测试和训练），代码也直观。
dataset = torch.utils.data.ConcatDataset([trainset, testset])
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
dataset_image, dataset_label = next(iter(dataset_loader))

'''
dataset_utils.py
'''
# v2: use ordered list to choose client, abandoned
num_client = 10
num_class_client = 2
idx_sample_client = [[] for _ in range(num_client)]
idx_client = np.array(range(num_client))
cnt_class_client = np.array([num_class_client for _ in range(num_client)])
available_client = np.array(range(num_client))

client_info = np.stack((cnt_class_client, available_client, idx_sample_client), axis=1)

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

def assign_data(dataset, partition):
    # replaced by subset
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