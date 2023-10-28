import random, os, pickle, json
import torch
import numpy as np
from tqdm import tqdm

from data.utils.setting import par_dict
from data.utils.datasets import DatasetDict, CustomSubset
from src.utils.setting import *
from src.utils.train_utils import get_best_device, Logger
from src.models.models import ModelDict
from src.client.fedavg import Client


class FedAvgServer(object):
    def __init__(self, datasets, models, args):
        self.args = args
        self.device = get_best_device(True)
        
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
        
        self.gdataset = datasets[0]
        self.gtestset = None
        self.gtestloader = None
        self.gtrans = None
        self.global_model = ModelDict[models[0]](
            self.gdataset, pretrained=self.args.pretrained
        ).to(self.device)
        self.global_loss = []
        self.global_acc = []
        self.trainer = Client
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.file_midname = f"{self.num_client}_{self.args.join_ratio}_{self.global_model.__class__.__name__}_{self.gdataset}_{self.args.global_round}"
        logfile = log_dir + self.file_midname + '.log'
        if os.path.exists(logfile):
            os.remove(logfile)
        self.logger = Logger(logfile)
        self.logger.log(f"Experiment Arguments:\n {dict(self.args._get_kwargs())}")
    
    def load_testset(self, transform=None):
        self.gtrans = transform
        dataset = DatasetDict[self.gdataset](transform=self.gtrans)
        with open(os.path.join(par_dict[self.gdataset], "partition.pkl"), "rb") as f:
            partition = pickle.load(f)
        test_indices = np.concatenate([client['test'] for client in partition])
        self.gtestset = CustomSubset(dataset, test_indices)
        self.gtestloader = torch.utils.data.DataLoader(
            self.gtestset, 
            batch_size=256, 
            shuffle=True, 
            # num_workers=self.args.num_workers,
            drop_last=False,
        )

    def round_train(self, round):
        self.selected_clients = self.client_sample_stream[round]
        self.client_model, self.client_weight = [], [] # cache clean
        for client_id in self.selected_clients: # real-world parallel, program-level sequential
            dataset, model = self.datasets[client_id], self.models[client_id]
            client = self.trainer(client_id, dataset, model, self.args, self.logger, self.device)
            client.receive(self.global_model)
            client.load_dataset(transform=self.gtrans)  # here apply global trans
            
            old_ls, old_acc = client.eval()
            train_ls, train_acc = client.train(save=False)
            new_ls, new_acc = client.eval()
            
            self.logger.log(f"client {client_id:02d}, (train) loss:{train_ls[-1]:.2f}|acc:{train_acc[-1]:.2f} (test) loss:{old_ls:.2f}->{new_ls:.2f}|acc:{old_acc:.2f}%->{new_acc:.2f}%")
            self.recive(client.upload())
    
    def train(self, save=True):
        for round in tqdm(range(self.args.global_round)):
            self.logger.log("-" * 32 + f"TRAINING EPOCH: {round + 1:2d}" + "-" * 32)
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
            for X, y in (self.gtestloader):
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
                client.load_dataset(transform=self.gtrans) # here apply global trans

                ls, acc = client.eval()
                cli_ls.append(ls)
                cli_acc.append(acc)
                
            self.logger.log(f"client_avg, (test)avg loss:{(sum(cli_ls)/len(cli_ls)):.2f}, avg accuracy:{(sum(cli_acc)/len(cli_acc)):.2f}%")
        
        return glo_ls, glo_acc, cli_ls, cli_acc