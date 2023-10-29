'''
参考Fl-bench之前的写法（FLBENCH： server全程只用一个client，trian的时候切换id实现获取对应的子集）
之前的写法：每个训练都会生成一个新的client
'''
import os, pickle
import torch
from torchvision import transforms
from tqdm import tqdm

from data.utils.datasets import DatasetDict, CustomSubset
from data.utils.setting import par_dict, MEAN, STD
from src.models.models import ModelDict
from src.utils.setting import state_dir


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
        
        self.trainset = None
        self.trainset = None
        self.trainloader = None
        self.testloader = None
        
        self.trained_epoch = 0 # maintain at server
        self.model = ModelDict[model](
            dataset, pretrained=self.args.pretrained
        ).to(device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.file_midname = f"{self.id}_{self.model.__class__.__name__}{'_ft_' if self.args.pretrained else '_'}{dataset}"

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[dataset], std=STD[dataset]),
            # transforms.Resize(224, antialias=True),
        ])
        self.dataset = DatasetDict[dataset](transform=transform) 
        with open(os.path.join(par_dict[dataset], "partition.pkl"), "rb") as f:
            partition = pickle.load(f)
        self.trainset = CustomSubset(self.dataset, partition[self.id]['train'])
        self.testset = CustomSubset(self.dataset, partition[self.id]['test'])
        
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
   
    def train(self, save=True):
        self.model.train()
        train_ls, train_acc = [], []
        for epoch in range(0, self.args.num_epochs): # tqdm optional, not recommend
            num_sample, ls, acc = 0, 0, 0
            # self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for X, y in (self.trainloader): # tqdm optional, not recommend
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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
        return self.model, len(self.trainset)

    def receive(self, model):
        model_state_dict = model.state_dict()
        client_state_dict = {}
        for key, value in model_state_dict.items():
            client_state_dict[key] = value.clone()
        self.model.load_state_dict(client_state_dict)
        
# --------------------------------------------------------
import random, os, pickle, json
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from data.utils.setting import par_dict, MEAN, STD
from data.utils.datasets import DatasetDict, CustomSubset
from src.utils.setting import *
from src.utils.train_utils import get_best_device, Logger
from src.models.models import ModelDict
from src.client.fedavg import Client


class FedAvgServer(object):
    def __init__(self, datasets, models, args):
        self.args = args
        self.device = get_best_device(True)
        
        # client setting
        self.num_client = self.args.num_client
        self.num_selected = max(1, int(self.args.join_ratio * self.num_client))
        self.client_sample_stream = [
            random.sample(range(self.num_client), self.num_selected) 
            for _ in range(self.args.global_round)
        ]
        self.datasets = random.choices(datasets, k=self.num_client) # ['minst', 'cifar10', ...]
        self.models = random.choices(models, k=self.num_client)  # ['lenet', 'vgg', ...]
        self.selected_clients = [] # dynmic at round i
        self.client_model = []
        self.client_weight = []
        self.client_loss = [] # track it mean always create many clients, quite expensive 
        self.client_acc = []

        # server setting: set [0] as default global model/dataset
        self.data_name = datasets[0]
        self.global_model = ModelDict[models[0]](
            self.data_name, pretrained=self.args.pretrained
        ).to(self.device)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[self.data_name], std=STD[self.data_name]),
            # transforms.Resize(224, antialias=True),
        ])
        self.global_dataset = DatasetDict[self.data_name](transform=transform)
        with open(os.path.join(par_dict[self.data_name], "partition.pkl"), "rb") as f:
            partition = pickle.load(f)
        test_indices = np.concatenate([client['test'] for client in partition])
        self.testset = CustomSubset(self.global_dataset, test_indices)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, 
            batch_size=256, 
            shuffle=True, 
            # num_workers=self.args.num_workers,
            drop_last=False,
        )
        self.global_loss = []
        self.global_acc = []

        # train setting
        self.trainer = Client
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # info setting
        self.file_midname = f"{self.num_client}_{self.args.join_ratio}_{self.global_model.__class__.__name__}_{datasets[0]}_{self.args.global_round}"
        logfile = log_dir + self.file_midname + '.log'
        if os.path.exists(logfile):
            os.remove(logfile)
        self.logger = Logger(logfile)
        self.logger.log(f"Experiment Arguments:\n {dict(self.args._get_kwargs())}")

    def round_train(self, round):
        self.selected_clients = self.client_sample_stream[round]
        self.client_model, self.client_weight = [], [] # cache clean
        for client_id in self.selected_clients: # real-world parallel, program-level sequential
            dataset, model = self.datasets[client_id], self.models[client_id]
            client = self.trainer(client_id, dataset, model, self.args, self.logger, self.device)
            client.receive(self.global_model)
            
            old_ls, old_acc = client.eval()
            train_ls, train_acc = client.train(save=False)
            new_ls, new_acc = client.eval()
            
            self.logger.log(f"client {client_id:02d}, (train) loss:{train_ls[-1]:.2f}|acc:{train_acc[-1]:.2f} (test) loss:{old_ls:.2f}->{new_ls:.2f}|acc:{old_acc:.2f}%->{new_acc:.2f}%")
            self.recive(client.upload())
    
    def train(self, save=True):
        for round in tqdm(range(self.args.global_round)):
            self.logger.log("-" * 32 + f"TRAINING EPOCH: {round + 1:02d}" + "-" * 32)
            self.round_train(round)
            self.aggregate()
            self.evaluate()
        self.logger.log("-" * 35 + f"FINAL RESULT" + "-" * 35)
        self.evaluate(client_eval=True)
        self.logger.close_log()
        
        if save:
            self.save_state()

    def recive(self, upload):
        client_param, num_sample = upload
        self.client_model.append(client_param)
        self.client_weight.append(num_sample)
    
    def aggregate(self):
        averaged_state_dict = {}
        weights = torch.tensor(self.client_weight) / sum(self.client_weight)
        # 遍历每个模型，将参数值累加到 averaged_state_dict 中
        for w, model in zip(weights, self.client_model):
            for name, param in model.state_dict().items():
                if name not in averaged_state_dict:
                    averaged_state_dict[name] = param.clone() * w
                else:
                    averaged_state_dict[name] += param.clone() * w
        self.global_model.load_state_dict(averaged_state_dict)

    def save_state(self):
        torch.save({
            'trained_round': self.args.global_round,
            'model_state_dict': self.global_model.state_dict(),
            }, state_dir+self.file_midname+f'_{self.args.global_round}.pth')
        
    def load_state(self, ckptfile):
        ckpt = torch.load(ckptfile)
        self.global_model.load_state_dict(ckpt['model_state_dict'])
        self.trained_epoch = ckpt['trained_round']
        # print(f'loading checkpoint!')

    def evaluate(self, client_eval=False):
        glo_ls, glo_acc = 0, 0
        cli_ls, cli_acc = [], []

        self.global_model.eval()
        num_samples, ls, acc = 0, 0, 0
        with torch.no_grad():
            for X, y in (self.testloader):
                X, y = X.to(self.device), y.to(self.device)
                output = self.global_model(X)
                loss = self.criterion(output, y)
                
                num_samples += y.size(0)
                ls += loss.item() * num_samples # reduction is mean by default
                acc += (torch.argmax(output.detach(), dim=1) == y).sum().item()
                
        glo_ls = ls / num_samples
        glo_acc = acc / num_samples * 100
        self.logger.log(f'global, (test) loss:{glo_ls:.2f}|acc:{glo_acc:.2f}%')
        
        if client_eval:
            for client_id in range(self.num_client):
                dataset, model = self.datasets[client_id], self.models[client_id]
                client = self.trainer(client_id, dataset, model, self.args, self.logger, self.device)
                client.receive(self.global_model)  # here apply global model

                ls, acc = client.eval()
                cli_ls.append(ls)
                cli_acc.append(acc)
                
            self.logger.log(f"client_avg, (test)avg loss:{(sum(cli_ls)/len(cli_ls)):.2f}, avg accuracy:{(sum(cli_acc)/len(cli_acc)):.2f}%")
        
        return glo_ls, glo_acc, cli_ls, cli_acc