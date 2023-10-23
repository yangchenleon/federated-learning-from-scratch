import random
import torch, torchvision
from torchvision import transforms
from src.client.fedavg import Client
from src.models.models import ModelDict

trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, antialias=True),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])
])

class FedAvgServer(object):
    def __init__(self, datasets, models, num_client, args):
        self.num_client = num_client
        self.args = args

        self.num_selected = max(1, int(self.args.join_ratio * self.num_client))
        self.client_sample_stream = [
            random.sample(self.train_clients, self.num_selected) 
            for _ in range(self.args.global_round)
        ]
        self.selected_clients = [] # at round i

        self.datasets = random.choices(datasets, num_client)  # ['minst', 'cifar10', ...]
        self.models = random.choices(models, num_client)  # ['lenet', 'vgg', ...]
        self.trainer = Client

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.global_model = ModelDict[models[0]](
            datasets[0], pretrained=self.args.pretrained
        ).to(device)
        self.client_model = [None for _ in range(self.num_selected)]
        self.client_weight = [None for _ in range(self.num_selected)]

    def round_train(self, round):
        self.selected_clients = self.client_sample_stream[round]
        for client_id in self.selected_clients: # real-world parallel, program-level sequential
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dataset, model = self.datasets[client_id], self.models[client_id]
            client = self.trainer(client_id, dataset, model, self.args, None, device)
            client.load_dataset(transform=trans)
            client.train(self.args.local_epoch)
            self.recive(client.upload())

    def recive(self, upload):
        id, client_param, num_sample = upload
        self.client_model[id] = client_param
        self.client_weight[id] = num_sample
    
    def aggregate(self):
        weights = torch.tensor(self.client_weight) / sum(self.client_weight)
        for glo_layer, cli_layer in zip(self.global_model.parameter(), zip(*self.client_model)):
            glo_layer.data = (torch.stack(cli_layer, dim=-1) * weights).sum(dim=-1)
        self.model.load_state_dict(self.global_params_dict, strict=False)
