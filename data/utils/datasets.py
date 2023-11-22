import os
import pandas as pd
import numpy as np
from PIL import Image
import torch, torchvision
from torch.utils.data.dataset import Dataset, Subset
from torchvision.transforms.functional import pil_to_tensor

from data.utils.setting import data_dict

class CustomDataset(Dataset):
    def __init__(self, transform):
        self.data = None
        self.targets = None
        self.classes = None
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        # not that data here is torchTensor
        data = np.array(data) # reverse to apply ToTensor() and other transforms
        if self.transform is not None:
            data = self.transform(data)
        return data, targets

class CustomSubset(Subset):
    # to access the subset, a whole dataset is required
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.targets = dataset.targets[indices]
        self.data = dataset.data[indices]
        self.classes = dataset.classes
        self.subset_transform = subset_transform
    # subset_transfrom 涉及__getitem__，其又涉及dataloader
    # 理论上直接return transform(self.dataset[indices[idx]])
    # 或者trans(self.data.numpy()[idx])可以了
    # 但是不确定还要验证，主要也用不上，暂不折腾
    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        # not that data here is torchTensor
        data = np.array(data) # reverse to apply ToTensor() and other transforms
        if self.subset_transform is not None:
            data = self.subset_transform(data)
        else:
            data = self.dataset.transform(data)
        return data, targets

class CustomMNIST(CustomDataset):
    def __init__(self, root=data_dict['mnist'], transform=None):
        super().__init__(transform)
        self.root = root

        trainset = torchvision.datasets.MNIST(root, train=True, download=True)
        testset = torchvision.datasets.MNIST(root, train=False, download=True)
        # here actually trainset.data is already a tensor, no need to torch.tensor()
        train_data = torch.Tensor(trainset.data).unsqueeze(dim=3).repeat([1, 1, 1, 3]) # add channel dim to apply most model
        test_data = torch.Tensor(testset.data).unsqueeze(dim=3).repeat([1, 1, 1, 3]) 
        train_targets = torch.Tensor(trainset.targets).squeeze()
        test_targets = torch.Tensor(testset.targets).squeeze()

        # dataset = torch.utils.data.ConcatDataset([trainset, testset])
        # dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False) # requrire add transform=Transform.toTensor()
        # self.data, self.target = next(iter(dataset_loader))

        self.data = torch.cat([train_data, test_data], dim=0) # uint8
        self.targets = torch.cat([train_targets, test_targets], dim=0)
        self.classes = trainset.classes

class CustomFashionMNIST(CustomDataset):
    def __init__(self, root=data_dict['fashion'], transform=None):
        super().__init__(transform)
        self.root = root

        trainset = torchvision.datasets.FashionMNIST(root, train=True, download=True)
        testset = torchvision.datasets.FashionMNIST(root, train=False, download=True)
        # here actually trainset.data is already a tensor, no need to torch.tensor()
        train_data = torch.Tensor(trainset.data).unsqueeze(dim=3).repeat([1, 1, 1, 3]) # add channel dim
        test_data = torch.Tensor(testset.data).unsqueeze(dim=3).repeat([1, 1, 1, 3])
        train_targets = torch.Tensor(trainset.targets).squeeze()
        test_targets = torch.Tensor(testset.targets).squeeze()

        self.data = torch.cat([train_data, test_data], dim=0)
        self.targets = torch.cat([train_targets, test_targets], dim=0)
        self.classes = trainset.classes

class CustomCIFAR10(CustomDataset):
    def __init__(self, root=data_dict['cifar10'], transform=None):
        super().__init__(transform)
        self.root = root

        # first time remember to switch True download
        trainset = torchvision.datasets.CIFAR10(root, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root, train=False, download=True)
        
        # here actually trainset.data is already a tensor, no need to torch.tensor()
        train_data = torch.Tensor(trainset.data).permute([0, 1, 2, 3]) # 0 -1 1 2 if dont use ToTensor()
        test_data = torch.Tensor(testset.data).permute([0, 1, 2, 3]) 
        train_targets = torch.Tensor(trainset.targets).long().squeeze()
        test_targets = torch.Tensor(testset.targets).long().squeeze()

        self.data = torch.cat([train_data, test_data], dim=0)
        self.targets = torch.cat([train_targets, test_targets], dim=0)
        self.classes = trainset.classes

class CustomEMNIST(CustomDataset):
    def __init__(self, root=data_dict['emnist'], transform=None):
        super().__init__(transform)
        self.root = root

        # EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
        # EMNIST ByMerge: 814,255 characters. 47 unbalanced classes.
        # EMNIST Balanced: 131,600 characters. 47 balanced classes.
        # EMNIST Letters: 145,600 characters. 26 balanced classes.
        # EMNIST Digits: 280,000 characters. 10 balanced classes.
        # EMNIST MNIST: 70,000 characters. 10 balanced classes.
        # default balanced: low data volumn but inlcude include all classes
        trainset = torchvision.datasets.EMNIST(root, train=True, download=True, split='balanced')
        testset = torchvision.datasets.EMNIST(root, train=False, download=True, split='balanced')
        
        # here actually trainset.data is already a tensor, no need to torch.tensor()
        train_data = torch.Tensor(trainset.data).unsqueeze(dim=3).repeat([1, 1, 1, 3]) # add channel dim
        test_data = torch.Tensor(testset.data).unsqueeze(dim=3).repeat([1, 1, 1, 3])
        train_targets = torch.Tensor(trainset.targets).long().squeeze()
        test_targets = torch.Tensor(testset.targets).long().squeeze()

        self.data = torch.cat([train_data, test_data], dim=0)
        self.targets = torch.cat([train_targets, test_targets], dim=0)
        self.classes = trainset.classes

class CustomTinyImageNet(CustomDataset):
    def __init__(self, root=data_dict['tinyimage'], transform=None):
        super().__init__(transform)

        self.classes = pd.read_table(
            root + "tiny-imagenet-200/wnids.txt", sep="\t", engine="python", header=None
        )[0].tolist()
        
        if not os.path.isfile(root + "data.pt") or not os.path.isfile(root + "targets.pt"):
            mapping = dict(zip(self.classes, list(range(len(self.classes)))))
            data = []
            targets = []
            train_dir  = f"{root}/tiny-imagenet-200/train/"
            for cls in os.listdir(train_dir):
                for img_name in os.listdir(f"{train_dir}/{cls}/images/"):
                    img = pil_to_tensor(Image.open(f"{train_dir}/{cls}/images/{img_name}")).float()
                    if img.shape[0] == 1: # dont know why some image has only one channel
                        img = torch.expand_copy(img, [3, 64, 64])
                    data.append(img.permute([1, 2, 0])) # permute back to align with other dataset
                    targets.append(mapping[cls])

            table = pd.read_table(
                root + "tiny-imagenet-200/val/val_annotations.txt", sep="\t", engine="python", header=None)
            test_classes = dict(zip(table[0].tolist(), table[1].tolist()))
            test_dir  = f"{root}/tiny-imagenet-200/val/"
            for img_name in os.listdir(f"{test_dir}/images"):
                img = pil_to_tensor(Image.open(f"{test_dir}/images/{img_name}")).float()
                if img.shape[0] == 1:
                    img = torch.expand_copy(img, [3, 64, 64])
                data.append(img.permute([1, 2, 0])) # permute back to align with other dataset
                targets.append(mapping[test_classes[img_name]])
            torch.save(torch.stack(data), root + "data.pt")
            torch.save(torch.tensor(targets, dtype=torch.long), root + "targets.pt")

        self.data = torch.load(root + "data.pt")
        self.targets = torch.load(root + "targets.pt")
        
class CustomMNISTM(CustomDataset):
    def __init__(self, root=data_dict['mnistm'], transform=None):
        super().__init__(transform)
        self.root = root
        
        # download it first https://github.com/liyxi/mnist-m/releases/tag/data 
        trainset = torch.load(os.path.join(self.root, 'mnist_m_train.pt'))
        testset = torch.load(os.path.join(self.root, 'mnist_m_test.pt'))
        
        # here actually trainset.data is already a tensor, no need to torch.tensor()
        train_data = torch.Tensor(trainset[0]).permute([0, 1, 2, 3]) # add channel dim
        test_data = torch.Tensor(testset[0]).permute([0, 1, 2, 3])
        train_targets = torch.Tensor(trainset[1]).long().squeeze()
        test_targets = torch.Tensor(testset[1]).long().squeeze()

        self.data = torch.cat([train_data, test_data], dim=0)
        self.targets = torch.cat([train_targets, test_targets], dim=0)
        self.classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

DatasetDict = {
    'mnist': CustomMNIST, 
    'fashion': CustomFashionMNIST,
    'cifar10': CustomCIFAR10, 
    'emnist': CustomEMNIST,
    'tinyimage': CustomTinyImageNet,
    'mnistm': CustomMNISTM,
}