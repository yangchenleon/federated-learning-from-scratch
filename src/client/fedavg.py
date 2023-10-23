import os, pickle, sys
import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F

sys.path.append('')
from data.utils.datasets import DatasetDict, CustomSubset
from src.models.models import ModelDict
from data.utils.setting import par_dict, data_dict

save_base = 'results/checkpoints/'

class Client(object):
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
        self.dataset = dataset
        self.model = ModelDict[model](dataset).to(device)

        self.trainset = None
        self.trainset = None
        self.trainloader = None
        self.testloader = None
        
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def load_dataset(self, transform=None, target_transform=None):
        '''
        default data_path is fixed in datasets, only set partition dir
        read partition and load train/test dataset
        '''
        dataset = DatasetDict[self.dataset](root=data_dict[self.dataset], transform=transform, target_transform=target_transform)
        with open(os.path.join(par_dict[self.dataset], "partition.pkl"), "rb") as f:
            partition = pickle.load(f)
        self.trainset = CustomSubset(dataset, partition[self.id]['train'])
        self.testset = CustomSubset(dataset, partition[self.id]['test'])
        
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
        return self.trainset, self.testset

    def train(self, startckpt=None, save=True):
        self.model.train()
        start, end = 0, self.args.num_epochs
        if startckpt is not None:
            start = self.load_state(startckpt)
        for epoch in range(start, end):
            ls, acc = 0, 0
            # self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for X, y in tqdm(self.trainloader):
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc += (torch.argmax(output.detach(), dim=1) == y).sum()
                ls += loss.item() * y.size(0)
            print(f'epoch:{epoch+1}  train loss:{ls/len(self.trainset):.3f}, train accuracy:{acc/len(self.trainset)*100:.2f}%')
        if save:
            self.save_state(save_base)
            
        # return self.model.state_dict()
    def eval(self):
        self.model.eval()
        num_samples,ls, acc = 0, 0, 0
        preds, trues = [], []
        # torch.no_grad() 上下文管理器是用于关闭梯度跟踪的，而 detach() 方法是用于分离张量的。在推断阶段，使用 torch.no_grad() 可以关闭梯度跟踪，而不需要使用 detach() 方法。在训练阶段，如果需要保留梯度信息，可以使用 detach() 方法来分离张量。
        with torch.no_grad():
            for X, y in tqdm(self.testloader):
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(self.model(X), y)
                
                num_samples += y.size(0)
                ls += loss.item() * num_samples # reduction is mean by default
                acc += (torch.argmax(output.detach(), dim=1) == y).sum()
                
                preds.append(output.detach().cpu().numpy())
                trues.append(F.one_hot(y.to(torch.int64), num_classes=len(self.testset.classes)).detach().cpu().numpy())
                # trues.append(label_binarize((y.detach().cpu().numpy()), classes=np.arange(len(self.dataset.num_class)))) # not useful, the classes part

        preds, trues = np.concatenate(preds), np.concatenate(trues)
        auc = metrics.roc_auc_score(trues, preds, average='macro')
        acc = acc / num_samples * 100
        ls = ls / num_samples
        print(f'test loss:{ls:.3f}, test accuracy:{acc:.2f}%')
        
        return acc, auc
    
    def save_state(self, path):
        ckpt_name = f"{self.id}_{self.dataset}_{self.model.__class__.__name__}_{self.args.num_epochs}.pth"

        torch.save({
            'trained_epoch': self.args.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path + ckpt_name)
    
    def load_state(self, ckptfile):
        ckpt = torch.load(ckptfile)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # print(f'loading checkpoint!')
        self.model.eval()

        return ckpt['trained_epoch']