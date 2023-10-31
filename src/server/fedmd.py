import torch

from src.server.fedavg import FedAvgServer
from src.client.fedmd import FedMDClient

'''
'''


class FedMDServer(FedAvgServer):
    def __init__(self, datasets, models, args):
        super().__init__(datasets, models, args)

        self.client_score = []
        self.consensus = None # in this setting, do not exist a global model, to initial a conensus for simplicity
        self.client = FedMDClient(0, self.data_name, self.model_name, self.args, self.logger, self.device)

    def round_train(self, round):
        self.client_score = []
        self.selected_clients = self.client_sample_stream[round]
        for client_id in self.selected_clients: # real-world parallel, program-level sequential
            dataset, model = self.datasets[client_id], self.models[client_id]
            self.client.switch(client_id, model, dataset)
            self.client.receive(self.distribute(client_id)) # global model here, client model if pFL
            
            old_ls, old_acc = self.client.eval()
            digest_ls, train_ls, train_acc = self.client.train(save=False)
            new_ls, new_acc = self.client.eval()
            self.recive(client_id, self.client.upload())
            
            self.logger.log(f"client {client_id:02d}, (train) dg:{digest_ls[-1]:.2f}|loss:{train_ls[-1]:.2f}|acc:{train_acc[-1]:.2f} (test) loss:{old_ls:.2f}->{new_ls:.2f}|acc:{old_acc:.2f}%->{new_acc:.2f}%")
    
    def recive(self, client_id, upload):
        client_score, res_upload = upload
        self.client_score.append(client_score)
        super().recive(client_id, res_upload)
    
    def distribute(self, client_id):
        package = (self.all_client_model[client_id], self.consensus)
        return package

    def aggregate(self):
        consensus = []
        for scores in zip(*self.client_score):
            consensus.append(torch.stack(scores, dim=-1).mean(dim=-1))
        self.consensus = consensus