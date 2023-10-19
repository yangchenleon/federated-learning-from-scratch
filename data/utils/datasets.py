import torch, torchvision
from torch.utils.data.dataset import Dataset, Subset


class CustomDataset(Dataset):
    def __init__(self):
        self.data = None
        self.targets = None
        self.classes = None
        self.data_transform = None
        self.target_transform = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        # not that data is usaully in its origin form, e.g. PIL, like 
        # img = Image.fromarray(img.numpy(), mode="L") in torchvision MINST, 
        # while in the later code, data and targets are already both tensor
        if self.data_transform is not None:
            data = self.data_transform(data)
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
        
        if self.subset_transform:
            x = self.subset_transform(x)
      
        return x, y   
    
    def __len__(self): 
        return len(self.indices)

class CustomMNIST(CustomDataset):
    def __init__(self, root='data', transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform 

        trainset = torchvision.datasets.MNIST(root, train=True, download=True)
        testset = torchvision.datasets.MNIST(root, train=False, download=True)
        # here actually trainset.data is already a tensor, no need to torch.tensor()
        train_data = trainset.data.unsqueeze(dim=1)
        test_data = testset.data.unsqueeze(dim=1)
        train_targets = trainset.targets.squeeze()
        test_targets = testset.targets.squeeze()

        self.data = torch.cat([train_data, test_data], dim=0)
        self.targets = torch.cat([train_targets, test_targets], dim=0)
        self.classes = trainset.classes

DatasetDict = {'mnist': CustomMNIST, 'cifar10': torchvision.datasets.CIFAR10, 'cifar100': torchvision.datasets.CIFAR100, 'fashionmnist': torchvision.datasets.FashionMNIST}