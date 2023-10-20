import os, pickle, sys
import torch
from tqdm import tqdm

sys.path.append('')
from data.utils.datasets import DatasetDict, CustomSubset
from models.models import ModelDict
from data.utils.setting import par_dict, data_dict

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
            # num_workers=self.args.num_workers
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            # num_workers=self.args.num_workers
        )
        return self.trainset, self.testset
    
    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            ls = 0
            # self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for X, y in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                loss = self.criterion(self.model(X), y)
                loss.backward()
                self.optimizer.step()
                ls += loss.item()
            print(ls/len(self.trainloader))
            
        # return self.model.state_dict()
    def eval(self):
        self.model.eval()
        X, y = next(iter(self.testloader))
        with torch.no_grad():
            preds = self.model(X).argmax(axis=1)
        
        print(preds, f'{sum(preds==y)/len(y):2f}')