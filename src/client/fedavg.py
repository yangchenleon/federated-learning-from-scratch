import os, pickle
import torch
from torchvision import transforms
from tqdm import tqdm

from data.utils.datasets import DatasetDict, CustomSubset
from data.utils.setting import par_dict, MEAN, STD
from src.models.models import ModelDict
from src.utils.setting import state_dir


class FedAvgClient(object):
    '''
    also as FedAvgClient
    '''
    def __init__(self, client_id, dataset, model, args, logger, device):
        # here dataset is actully dataset_name, using dict to acquire
        # because usually do not use class as parameter
        # beside, a clint only need a subset, only train/test set, not whole
        self.args = args
        self.device = device
        self.logger = logger
        self.id = client_id
        self.dataset_name = dataset
        
        self.trainset = None
        self.trainset = None
        self.trainloader = None
        self.testloader = None
        
        self.trained_epoch = 0 # maintain at server
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # consider all same dataset setting, so only need to init once, consider move to switch if mutli-dataset setting
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=MEAN[dataset], std=STD[dataset]),
            # transforms.Resize(100, antialias=True),
        ])
        self.dataset = DatasetDict[dataset](transform=transform) 
        with open(os.path.join(par_dict[dataset], "partition.pkl"), "rb") as f:
            self.partition = pickle.load(f)
        self.model = None
        
    def train(self, save=True):
        self.model.train()
        train_ls, train_acc = [], []
        for epoch in range(0, self.args.num_epochs): # tqdm optional, not recommend
            num_sample, ls, acc = 0, 0, 0
            for X, y in (self.trainloader): # tqdm optional, not recommend
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

                acc += (torch.argmax(output.detach(), dim=1) == y).sum().item()
                ls += loss.item() * y.size(0)
                num_sample += len(y) # same as y.size(0), y.shape[0]
            
            train_ls.append(ls/num_sample)
            train_acc.append(acc/num_sample*100)
            # self.logger.log(f'{self.id}_epoch:{epoch+1}  train loss:{ls/num_sample:.3f}, train accuracy:{acc/num_sample*100:.2f}%')
        
        if save:
            self.save_state()

        return train_ls, train_acc
            
    def eval(self):
        self.model.eval()
        num_samples, ls, acc = 0, 0, 0
        with torch.no_grad():
            for X, y in (self.testloader):
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                
                num_samples += y.size(0)
                ls += loss.item() * num_samples # reduction is mean by default
                acc += (torch.argmax(output.detach(), dim=1) == y).sum().item()
                
        acc = acc / num_samples * 100
        ls = ls / num_samples
        # print(f'client{self.id}, test loss:{ls:.3f}, test accuracy:{acc:.2f}%')
        
        return ls, acc
    
    def save_state(self):
        torch.save({
            'trained_epoch': self.trained_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, state_dir+self.file_midname+f'_{self.trained_epoch}.pth'
        )
    
    def load_state(self, ckptfile):
        ckpt = torch.load(ckptfile)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.trained_epoch = ckpt['trained_epoch']
        # print(f'loading checkpoint!')
        self.model.eval()
    
    def upload(self):
        return self.id, self.model, len(self.trainset)

    def receive(self, package):
        model = package
        if model is None:
            # global model initializing, same as self.model = self.model
            return
        model_state_dict = model.state_dict()
        client_state_dict = {}
        for key, value in model_state_dict.items():
            client_state_dict[key] = value.clone()
        self.model.load_state_dict(client_state_dict)
        
    def switch(self, client_id, model, dataset):
        self.id = client_id
        self.trainset = CustomSubset(self.dataset, self.partition[self.id]['train'])
        self.testset = CustomSubset(self.dataset, self.partition[self.id]['test'])
        
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            # num_workers=self.args.num_workers,
            drop_last=True, # When the current batch size is 1, the batchNorm2d modules in the model would raise error. So the latent size 1 data batches are discarded.
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            # num_workers=self.args.num_workers,
            drop_last=False,
        )
        # only in customsubse and in uint8, use [index] to apply transform
        # print(self.trainset.data.shape, self.trainset.data.dtype)
        # print(next(iter(self.trainloader))[0].shape, next(iter(self.trainloader))[0].dtype)

        self.model_name = model
        self.model = ModelDict[self.model_name](
            self.dataset_name, pretrained=self.args.pretrained
        ).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.args.lr, 
            momentum=self.args.momentum, 
            weight_decay=self.args.weight_decay
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.file_midname = f"{self.id}_{self.model_name}{'_ft_' if self.args.pretrained else '_'}{self.dataset_name}"