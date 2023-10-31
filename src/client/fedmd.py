import random
import torch
from torchvision import transforms

from src.client.fedavg import FedAvgClient
from data.utils.datasets import DatasetDict, CustomSubset
from data.utils.setting import MEAN, STD


class FedMDClient(FedAvgClient):
    def __init__(self, client_id, dataset, model, args, logger, device):
        super().__init__(client_id, dataset, model, args, logger, device)

        self.score_critertion = torch.nn.MSELoss()
        self.consensus = None

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=MEAN[self.args.public_dataset], 
                std=STD[self.args.public_dataset]
            ),
            # transforms.Resize(224, antialias=True),
        ])
        public_dataset = DatasetDict[self.args.public_dataset](transform=transform)
        self.public_dataset = CustomSubset(public_dataset, random.sample(list(range(len(public_dataset))), self.args.public_size))
        self.publicloader = torch.utils.data.DataLoader(
            self.public_dataset, 
            self.args.public_batch_size, 
            shuffle=False,
            drop_last=True,
        )
    
    def upload(self):
        self.model.eval()
        score = []
        with torch.no_grad():
            for X, _ in self.publicloader:
                X = X.to(self.device)
                score.append(self.model(X).detach())
        return score, (self.model, len(self.trainset))

    def receive(self, package):
        model, consensus = package
        if consensus is None:
            self.consensus = torch.stack(self.upload()[0])
        else:
            self.consensus = consensus
        super().receive(model)

    def digest(self):
        self.model.train()
        num_samples, ls = 0, 0
        digest_ls = []
        for _ in range(self.args.num_digest_epoch):
            for i, data in enumerate(self.publicloader):
                X, y = data[0].to(self.device), data[1].to(self.device)
                score = self.model(X)
                loss = self.score_critertion(score, self.consensus[i])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_samples += y.size(0)
                ls += loss.item() * num_samples
            
            digest_ls.append(ls/num_samples)
        return digest_ls

    def train(self, save=True):
        digest_ls = self.digest()
        train_ls, train_acc = super().train(save)
        return digest_ls, train_ls, train_acc