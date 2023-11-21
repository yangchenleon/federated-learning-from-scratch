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
from src.client.fedavg import FedAvgClient


class FedAvgServer(object):
    def __init__(self, datasets, models, args):
        self.algo_name = type(self).__name__[:-6]
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
        self.client_loss = [] # track it mean always create many clients, quite expensive 
        self.client_acc = []
        self.all_client_weight = [None for _ in range(self.num_client)]
        self.all_client_model = [None for _ in range(self.num_client)]

        # server setting: set [0] as default global model/dataset
        self.data_name = datasets[0]
        self.model_name = models[0]
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[self.data_name], std=STD[self.data_name]),
            # transforms.Resize(224, antialias=True),
        ])
        self.global_dataset = DatasetDict[self.data_name](transform=transform)
        self.global_model = ModelDict[self.model_name](
            num_classes=len(self.global_dataset.classes)
        ).to(self.device)
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
               
        # info setting
        self.file_midname = f"{self.num_client}_{self.args.partition}_{self.args.join_ratio}_{self.model_name}_{self.data_name}_{self.args.global_round}"
        logfile = log_dir + self.algo_name + '//' + self.file_midname + '.log'
        if not os.path.exists(os.path.dirname(logfile)):
            os.mkdir(log_dir + self.algo_name)
        self.logger = Logger(logfile)
        self.logger.log(f"Experiment Arguments:\n {dict(self.args._get_kwargs())}")
        self.logger.log(f"client datasets: {self.datasets}")
        self.logger.log(f"client models: {self.models}")

        # train setting
        self.client = FedAvgClient(0, self.data_name, self.model_name, self.args, self.logger, self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def round_train(self):
        for client_id in self.selected_clients: # real-world parallel, program-level sequential
            dataset, model = self.datasets[client_id], self.models[client_id]
            # 每一次训练都会生成一个client，大佬的做法是同一个client对象，但是数据集还是统一，训练输入给client编号，根据编号重新设置训练和测试子集、dataloader。多了创建对象、重新读取dataset、partition的步骤，考虑到暂时可优化反复读数据集的情况，可修改，改了后仍可拓展为多dataset的情况。
            # client = self.trainer(client_id, dataset, model, self.args, self.logger, self.device)
            self.client.switch(client_id, model, dataset)
            self.client.receive(self.distribute(client_id)) # global model here, client model if pFL
            
            old_ls, old_acc = self.client.eval()
            train_ls, train_acc = self.client.train(save=False)
            new_ls, new_acc = self.client.eval()
            self.recive(self.client.upload())
            
            self.logger.log(f"client {client_id:02d}, (train) loss:{train_ls[-1]:.2f}|acc:{train_acc[-1]:.2f}% (test) loss:{old_ls:.2f}->{new_ls:.2f}|acc:{old_acc:.2f}%->{new_acc:.2f}%")
    
    def train(self, save=True):
        for round in tqdm(range(self.args.global_round)):
            self.logger.log("-" * 32 + f"TRAINING EPOCH: {round + 1:02d}" + "-" * 32)
            self.selected_clients = self.client_sample_stream[round]
            self.round_train()
            self.aggregate()
            self.evaluate()
        self.logger.log("-" * 35 + f"FINAL RESULT" + "-" * 35)
        self.evaluate()
        self.logger.close_log()
        
        if save:
            self.save_state()

    def recive(self, upload):
        id, model, num_sample = upload
        self.all_client_weight[id] = num_sample
        self.all_client_model[id] = model
    
    def distribute(self, client_id):
        package = self.all_client_model[client_id]
        return package
    
    def aggregate(self):
        averaged_state_dict = {}
        weights = [self.all_client_weight[i] for i in self.selected_clients]
        weights = torch.tensor(weights) / sum(weights)
        models = [self.all_client_model[i] for i in self.selected_clients]

        for w, model in zip(weights, models):
            for name, param in model.state_dict().items():
                if name not in averaged_state_dict:
                    averaged_state_dict[name] = param.clone() * w
                else:
                    averaged_state_dict[name] += param.clone() * w
        self.global_model.load_state_dict(averaged_state_dict)
        for i in range(self.num_client): # typical way of fedavg
            self.all_client_model[i] = self.global_model

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

    def evaluate(self):
        glo_ls, glo_acc = 0, 0
        cli_ls, cli_acc = [], []

        if self.args.global_eval:
            self.global_model.eval()
            num_samples, ls, acc = 0, 0, 0
            with torch.no_grad():
                for X, y in (self.testloader):
                    X, y = X.to(self.device), y.to(self.device)
                    output = self.global_model(X)
                    loss = self.criterion(output, y)
                    
                    num_samples += y.size(0)
                    ls += loss.item() * num_samples # reduction is mean by default
                    acc += (torch.argmax(output.detach(), dim=1) == y).sum().item() # detach is actully meaningless here
                    
            glo_ls = ls / num_samples
            glo_acc = acc / num_samples * 100
            self.logger.log(f'global, (test) loss:{glo_ls:.2f}|acc:{glo_acc:.2f}%')
        
        if self.args.client_eval:
            for client_id in range(self.num_client):
                dataset, model = self.datasets[client_id], self.models[client_id]
                self.client.switch(client_id, model, dataset)
                self.client.receive(self.distribute(client_id)) # global model here, client model if pFL

                ls, acc = self.client.eval()
                cli_ls.append(ls)
                cli_acc.append(acc)
                
            self.logger.log(f"client_avg, (test)avg loss:{(sum(cli_ls)/len(cli_ls)):.2f}, avg accuracy:{(sum(cli_acc)/len(cli_acc)):.2f}%")
        
        return glo_ls, glo_acc, cli_ls, cli_acc