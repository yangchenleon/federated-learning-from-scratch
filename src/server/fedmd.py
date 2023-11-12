import torch

from src.server.fedavg import FedAvgServer
from src.client.fedmd import FedMDClient

'''
关于复现，少代码上没什么问题（如果public和priv的是一样的话，不说digest有没有提升，loss确实有降，至少不会影响自个训练，然后如果pub=cifar10，pri=fashion，digest就会影响本身的训练，明显就是这之间的差距太大了，upload聚合的结果本身就是错误，无法指导迁移）
文章说法是本地数据集用的FEMNIST（就是LEAF那个，虽然说non-iid是基于区分writer的更真实，但还是有点麻烦，不如直接用EMNIST自己分类区别不大，所以没做实验，主要是懒），公共的用MNSIT，这个相比较fashion和cifar10的差距小很多；第二组实验用的cifar10作为pub，ciar100的子集作为priv。第一都是手写数据，第二FEMNIST包含MNIST，有点接近我的做法，第二组同理。
在训练方面，首先在公共数据集上收敛，这其实回答了我用cifar10作为公共的数据集出现的问题，至少共享的知识是有效的，然后再在priv上训练（为了表现效果，这部就免了，以免起点太高），最后再知识共享。问题就在第一步，既然已经学习了pub，相当于提前看了参考答案，模型对数据已经有了基本判断，之后的共享的答案基本失去意义，甚至可能因为不够标准降低判断效果。
本来的目的是通过同步的公共认知协助本地的判断，但实际上不需要同步，自己的认知几乎就是共同认知，按照实验细节协助的数据比自己的还差，协助效果甚微。那么如果不做预训练，共识是错的没有方向的，甚至破坏模型学习能力。感觉模型能work主要还是本地训练，和digest一点关系没有；很佩服水论文的能力，明明就是各干各的，硬是通过digest的模式实现协作，套上了联邦的外壳。实验效果还是存疑，将预训练后pritest作为基线，协同效果有提升，但是最终小于单纯基于本地训练效果，iid情况下不应该出现这种情况才对，影响因素只有数据量和预训练的差别，预训练的影响不至于而且本地的效果应该会覆盖才对，数据量少训练轮次还低造成的（果然只train了20轮，privdata只有200条但是每种类都有）
另外要吐槽的是，emnist其实只有letters数据集，且仅采用前11个作为priv；另外，严格来说，没有遵行KD的范式，即softmax的logit作为知识，甚至直接用的不带softmax的结果，说没啥区别，直接用的-2层的输出，去除了-1层的softmax层的最终输出，关于模型输出为什么最后一层是sm又是另一个奇怪的问题了，毕竟argmax的结果是一样的。
也指引了两个方向，第一，缺少有效的共识，一种就是通过预训练（果然还是离不开这个），非要说本篇也算，不过相当于用本地预训练代替了通过logit学习的过程-作弊，欸，等等，不要预训练呢，找那个至少能预测对一些的慢慢学（好像是个方向）；第二的问题就是领域差距，效果无法训练的原因就是差别太大，能训练的原因就是因为在同一领域，那么解决办法就是域适应。
'''

class FedMDServer(FedAvgServer):
    def __init__(self, datasets, models, args):
        super().__init__(datasets, models, args)

        self.all_client_score = [None for i in range(self.num_client)]
        self.consensus = None # in this setting, do not exist a global model, to initial a conensus for simplicity
        self.client = FedMDClient(0, self.data_name, self.model_name, self.args, self.logger, self.device)

    def round_train(self):
        for client_id in self.selected_clients: # real-world parallel, program-level sequential
            dataset, model = self.datasets[client_id], self.models[client_id]
            self.client.switch(client_id, model, dataset)
            self.client.receive(self.distribute(client_id)) # global model here, client model if pFL
            
            old_ls, old_acc = self.client.eval()
            digest_ls, train_ls, train_acc = self.client.train(save=False)
            new_ls, new_acc = self.client.eval()
            self.recive(self.client.upload())
            
            self.logger.log(f"client {client_id:02d}, (train) dgs:{digest_ls[-1]:.2f}|loss:{train_ls[-1]:.2f}|acc:{train_acc[-1]:.2f} (test) loss:{old_ls:.2f}->{new_ls:.2f}|acc:{old_acc:.2f}%->{new_acc:.2f}%")
    
    def recive(self, upload):
        upload, client_score = upload
        id = upload[0]
        self.all_client_score[id] = client_score
        super().recive(upload)
    
    def distribute(self, client_id):
        package = (self.all_client_model[client_id], self.consensus)
        return package

    def aggregate(self):
        consensus = []
        client_score = [self.all_client_score[i] for i in self.selected_clients]
        for scores in zip(*client_score):
            consensus.append(torch.stack(scores, dim=-1).mean(dim=-1))
        self.consensus = consensus
    
    def train(self, save=True):
        self.logger.log("pretraining")
        for client_id in range(self.num_client):
            dataset, model = self.datasets[client_id], self.models[client_id]
            self.client.switch(client_id, model, dataset)
            self.client.pretain()
            self.recive(self.client.upload())
        return super().train(save)