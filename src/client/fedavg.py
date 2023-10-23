import os, pickle, sys, json
import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data.utils.datasets import DatasetDict, CustomSubset
from src.models.models import ModelDict
from data.utils.setting import par_dict, data_dict
from src.utils.setting import curve_dir, state_dir, figure_dir

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
        
        self.trainset = None
        self.trainset = None
        self.trainloader = None
        self.testloader = None
        
        self.trained_epoch = 0
        self.model = ModelDict[model](
            dataset, pretrained=self.args.pretrained
        ).to(device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.learn_curve = []
        self.file_midname = f"{self.id}_{self.model.__class__.__name__}{'_ft_' if self.args.pretrained else '_'}{self.dataset}"

    def load_dataset(self, transform=None, target_transform=None):
        '''
        default data_path is fixed in datasets, only set partition dir
        read partition and load train/test dataset
        '''
        dataset = DatasetDict[self.dataset](root=data_dict[self.dataset],transform=transform, target_transform=target_transform)
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
        if startckpt is not None:
            self.load_state(startckpt)
            with open(curve_dir+self.file_midname+f"_{self.trained_epoch}.json", 'r') as f:
                self.learn_curve = json.load(f)
        for epoch in range(self.trained_epoch, self.args.num_epochs):
            num_sample, ls, acc = 0, 0, 0
            # self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for X, y in tqdm(self.trainloader):
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc += (torch.argmax(output.detach(), dim=1) == y).sum().item()
                ls += loss.item() * y.size(0)
                num_sample += len(y) # same as y.size(0), y.shape[0]
            
                self.learn_curve.append((ls/num_sample, acc/num_sample*100))
            print(f'epoch:{epoch+1}  train loss:{ls/num_sample:.3f}, train accuracy:{acc/num_sample*100:.2f}%')
        if save:
            self.save_state()
            
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
    
    def save_state(self):
        torch.save({
            'trained_epoch': self.args.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, state_dir+self.file_midname+f'_{self.args.num_epochs}.pth')

        with open(curve_dir+self.file_midname+f'_{self.args.num_epochs}.json', 'w') as f:
            json.dump(self.learn_curve, f)
    
    def load_state(self, ckptfile):
        ckpt = torch.load(ckptfile)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.trained_epoch = ckpt['trained_epoch']
        # print(f'loading checkpoint!')
        self.model.eval()

        return ckpt['model_state_dict']
    
    def upload(self):
        # 1. with parameters()
        # return self.model.parameters() # iterate layer, .data to get tensor
        # 2. with state_dict()
        parameters, keys = [], [] # keys is acutally useless
        for name, param in self.model.state_dict(keep_vars=True).items(): # in case need gradient
            if param.requires_grad:
                parameters.append(param.detach().clone()) # .data equal to .detach()
                keys.append(name)
        return self.id, parameters, len(self.trainset)

    def draw_curve(self):
        with open(curve_dir+self.file_midname+f'_{self.args.num_epochs}.json', 'r') as f:
            self.learn_curve = json.load(f)
        train_loss, train_acc = zip(*self.learn_curve)
        fig, ax1 = plt.subplots()
        ax1.plot(train_loss, 'r-', label='train_loss')
        ax1.set_xlabel('X')
        ax1.set_ylabel('loss', color='r')
        ax1.tick_params('y', colors='r')
        ax2 = ax1.twinx()

        # 绘制第二个y轴对应的数据
        ax2.plot(train_acc, 'b-', label='train_acc')
        ax2.set_ylabel('accuracy', color='b')
        ax2.tick_params('y', colors='b')

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines])
        plt.savefig(figure_dir + self.file_midname+f'_{self.args.num_epochs}.jpg')
        plt.show()
    
        