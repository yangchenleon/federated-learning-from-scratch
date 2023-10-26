import random, os, pickle, json
import torch
import numpy as np
from tqdm import tqdm

from data.utils.setting import par_dict
from data.utils.datasets import DatasetDict, CustomSubset
from src.utils.setting import state_dir, curve_dir
from src.utils.model_utils import get_best_device
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
        
        self.gdataset = datasets[0]
        self.gtestset = None
        self.gtestloader = None
        self.gtrans = None
        self.global_model = ModelDict[models[0]](
            self.gdataset, pretrained=self.args.pretrained
        ).to(self.device)
        self.trainer = Client
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.file_midname = f"{self.num_client}_{self.args.join_ratio}_{self.global_model.__class__.__name__}_{self.gdataset}"
        self.log_info = [] # round, participants, acc(glo), loss(glo), acc(cli-avg), loss(cli-avg)
        self.learn_curve = []
    
    def load_testset(self, transform=None, target_transform=None):
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
            client = self.trainer(client_id, dataset, model, self.args, None, self.device)
            client.receive(self.global_model)
            client.load_dataset(transform=self.gtrans)  # here apply global trans
            
            old_ls, old_acc = client.eval()
            client.train(save=False)
            new_ls, new_acc = client.eval()
            
            print(f'client{client_id:02d}, test loss:{old_ls:.2f}->{new_ls:.2f}, test accuracy:{old_acc:.2f}%->{new_acc:.2f}%')
            self.recive(client.upload())
    
    def train(self, save=True):
        for round in tqdm(range(self.args.global_round)):
            self.round_train(round)
            self.aggregate()
            self.evaluate()
        
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
        
        with open(curve_dir+self.file_midname+f'_{self.args.global_round}.json', 'w') as f:
            json.dump(self.learn_curve, f)

    def load_state(self, ckptfile):
        ckpt = torch.load(ckptfile)
        self.global_model.load_state_dict(ckpt['model_state_dict'])
        self.trained_epoch = ckpt['trained_round']
        # print(f'loading checkpoint!')
        self.global_model.eval()

    def evaluate(self):
        self.global_model.eval()
        num_samples,ls, acc = 0, 0, 0
        with torch.no_grad():
            for X, y in (self.gtestloader):
                X, y = X.to(self.device), y.to(self.device)
                output = self.global_model(X)
                loss = self.criterion(output, y)
                
                num_samples += y.size(0)
                ls += loss.item() * num_samples # reduction is mean by default
                acc += (torch.argmax(output.detach(), dim=1) == y).sum().item()
                
        acc = acc / num_samples * 100
        ls = ls / num_samples
        self.learn_curve.append((ls, acc))
        print(f'global, test loss:{ls:.3f}, test accuracy:{acc:.2f}%')
        
        return ls, acc