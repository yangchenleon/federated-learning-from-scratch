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
    # to access the subset, a whole dataset is required
    def __init__(self, dataset, indices, subset_transform=None):
        # super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.targets = dataset.targets[indices]
        self.data = dataset.data[indices]
        self.classes = dataset.classes
        self.subset_transform = subset_transform

    def __getitem__(self, index):
        data, targets = self.dataset[index]
        # x, y = self.data[idx], self.targets[idx]
        # here use index to get, so fatherset transform is applied
        if self.subset_transform:
            x = self.subset_transform(x)
      
        return data, targets
    
    def __len__(self): 
        return len(self.indices)

class CustomMNIST(CustomDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(transform, target_transform)
        self.root = root

        trainset = torchvision.datasets.MNIST(root, train=True, download=True)
        testset = torchvision.datasets.MNIST(root, train=False, download=True)
        # here actually trainset.data is already a tensor, no need to torch.tensor()
        train_data = torch.Tensor(trainset.data).unsqueeze(dim=3).repeat([1, 1, 1, 3]) # add channel dim    
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
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(transform, target_transform)
        self.root = root

        trainset = torchvision.datasets.FashionMNIST(root, train=True, download=True)
        testset = torchvision.datasets.FashionMNIST(root, train=False, download=True)
        # here actually trainset.data is already a tensor, no need to torch.tensor()
        train_data = torch.Tensor(trainset.data).unsqueeze(dim=3) # add channel dim
        test_data = torch.Tensor(testset.data).unsqueeze(dim=3)
        train_targets = torch.Tensor(trainset.targets).squeeze()
        test_targets = torch.Tensor(testset.targets).squeeze()

        self.data = torch.cat([train_data, test_data], dim=0)
        self.targets = torch.cat([train_targets, test_targets], dim=0)
        self.classes = trainset.classes

class CustomCIFAR10(CustomDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(transform, target_transform)
        self.root = root

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


DatasetDict = {
    'mnist': CustomMNIST, 
    'fashion': CustomFashionMNIST,
    'cifar10': CustomCIFAR10, 
    'cifar100': torchvision.datasets.CIFAR100, 
}