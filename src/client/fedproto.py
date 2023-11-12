import copy
import numpy as np
import torch
from src.client.fedavg import FedAvgClient


class FedProtoClient(FedAvgClient):
    def __init__(self, client_id, dataset, model, args, logger, device):
        super().__init__(client_id, dataset, model, args, logger, device)
        
        self.global_prototype = None
        self.regular_loss = torch.nn.MSELoss()
    
    def upload(self):
        return super().upload(), self.cal_proto()
    
    def receive(self, package):
        package, global_prototype = package
        if global_prototype.__len__() == 0:
            self.global_prototype = self.cal_proto()[1]
        else:
            self.global_prototype = global_prototype
        return super().receive(package)

    def cal_proto(self):
        self.model.eval()
        prototype, class_cnt = {}, {}
        with torch.no_grad():
            for X, y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)
                batch_emb = self.base(X)
                for i, cls in enumerate(y.tolist()):
                    if cls not in prototype:
                        prototype[cls] = batch_emb[i].detach().to('cpu')
                        class_cnt[cls] = 1
                    else:
                        prototype[cls] += batch_emb[i].detach().to('cpu')
                        class_cnt[cls] += 1
        for i in class_cnt.keys():
            prototype[i] /= class_cnt[i]
        return class_cnt, prototype
    
    def train(self, save=True):
        self.model.train()
        train_ls, train_acc = [], []
        for epoch in range(0, self.args.num_epochs):
            num_sample, ls, acc = 0, 0, 0
            for X, y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)
                emb = self.base(X)
                output = self.classifier(emb)
               
                batch_glo_proto = []
                for cls in y.tolist():
                    batch_glo_proto.append(self.global_prototype[cls])
                glo_proto = torch.stack(batch_glo_proto).to(self.device)
                loss = self.criterion(output, y) + self.args.lamda * self.regular_loss(emb, glo_proto)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc += (torch.argmax(output.detach(), dim=1) == y).sum().item()
                ls += loss.item() * y.size(0)
                num_sample += len(y) # same as y.size(0), y.shape[0]
            
            train_ls.append(ls/num_sample)
            train_acc.append(acc/num_sample*100)

        if save:
            self.save_state()
        
        return train_ls, train_acc          

    def switch(self, client_id, model, dataset):
        super().switch(client_id, model, dataset)
        self.classifier = self.model.classifier
        if hasattr(self.model, 'base'):
            self.base = self.model.base
        else:
            self.base = self.model
            self.base.classifier = torch.nn.Identity()