import numpy as np
import torch, torchvision
from torch.utils.data.dataset import Dataset, Subset


class CustomDataset(Dataset):
    def __init__(self, transform, target_transform):
        self.data = None
        self.targets = None
        self.classes = None
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        # not that data here is torchTensor
        data = np.array(data) # reverse to apply ToTensor() and other transforms
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return data, targets

class CustomSubset(Subset):
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.targets = dataset.targets[indices]
        self.data = dataset.data[indices]
        self.classes = dataset.classes
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        # here use index to get, so fatherset transform is applied
        if self.subset_transform:
            x = self.subset_transform(x)
      
        return x, y   
    
    def __len__(self): 
        return len(self.indices)

class CustomMNIST(CustomDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(transform, target_transform)
        self.root = root

        trainset = torchvision.datasets.MNIST(root, train=True, download=True)
        testset = torchvision.datasets.MNIST(root, train=False, download=True)
        # here actually trainset.data is already a tensor, no need to torch.tensor()
        train_data = trainset.data.unsqueeze(dim=1)
        test_data = testset.data.unsqueeze(dim=1)
        train_targets = trainset.targets.squeeze()
        test_targets = testset.targets.squeeze()

        # dataset = torch.utils.data.ConcatDataset([trainset, testset])
        # dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False) # requrire add transform=Transform.toTensor()
        # self.data, self.target = next(iter(dataset_loader))

        self.data = torch.cat([train_data, test_data], dim=0)
        self.targets = torch.cat([train_targets, test_targets], dim=0)
        self.classes = trainset.classes

class CustomFashionMNIST(CustomDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(transform, target_transform)
        self.root = root

        trainset = torchvision.datasets.FashionMNIST(root, train=True, download=True)
        testset = torchvision.datasets.FashionMNIST(root, train=False, download=True)
        # here actually trainset.data is already a tensor, no need to torch.tensor()
        train_data = trainset.data.unsqueeze(dim=1)
        test_data = testset.data.unsqueeze(dim=1)
        train_targets = trainset.targets.squeeze()
        test_targets = testset.targets.squeeze()

        self.data = torch.cat([train_data, test_data], dim=0)
        self.targets = torch.cat([train_targets, test_targets], dim=0)
        self.classes = trainset.classes

DatasetDict = {
    'mnist': CustomMNIST, 
    'fashion': CustomFashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10, 
    'cifar100': torchvision.datasets.CIFAR100, 
}