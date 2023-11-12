import torch

from src.server.fedavg import FedAvgServer
from src.client.fedproto import FedProtoClient

class FedProtoServer(FedAvgServer):
    def __init__(self, datasets, models, args):
        super().__init__(datasets, models, args)

        self.all_client_prototype = [{} for i in range(self.num_client)]
        self.all_client_classcnt = [{} for i in range(self.num_client)]
        self.global_prototype = {}

        self.client = FedProtoClient(0, self.data_name, self.model_name, self.args, self.logger, self.device)

    def recive(self, upload):
        upload, proto = upload
        id = upload[0]
        self.all_client_prototype[id] = proto[1]
        self.all_client_classcnt[id] = proto[0]
        return super().recive(upload)

    def distribute(self, client_id):
        client_class_key = self.all_client_classcnt[client_id].keys()
        global_prototype = {key: self.global_prototype[key] for key in client_class_key}
        return self.all_client_model[client_id], global_prototype

    def aggregate1(self):
        sum_class_cnt = {}
        sum_prototype = {}
        for client_id in self.selected_clients:
            for cls, cnt in self.all_client_classcnt[client_id].items():
                if cls not in sum_class_cnt.keys():
                    sum_class_cnt[cls] = cnt
                else:
                    sum_class_cnt[cls] += cnt
            for cls, prototype in self.all_client_prototype[client_id].items():
                if cls not in sum_prototype.keys():
                    sum_prototype[cls] = prototype
                else:
                    sum_prototype[cls] += prototype
        for cls in self.all_client_classcnt[client_id].keys():
            sum_prototype[cls] /= sum_class_cnt[cls]
     
    # 原版有问题啊，甚至于论文也有问题啊，这个prototype应该是样本的平均表征，还要除以拥有客户数干啥。但是原版代码直接除以客户数又回来了，相当于来自n个客户的平均表征的再平均，回到了样本平均，但相当于对每个客户的表征权重是一样的，这个公式和论文又不一样了。唉，感觉论文的不太对。写了两个aggregation，按后者来吧

    def aggregate(self):
        agg_prototype = {}
        for client_id in self.selected_clients:
             for cls, prototype in self.all_client_prototype[client_id].items():
                if cls not in agg_prototype.keys():
                    agg_prototype[cls] = [prototype]
                else:
                    agg_prototype[cls].append(prototype)
        for cls in agg_prototype.keys():
            self.global_prototype[cls] = torch.mean(torch.stack(agg_prototype[cls]), dim=0)        