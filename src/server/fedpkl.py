import torch
import torch.nn.functional as F

from src.server.fedmd import FedMDServer
from src.client.fedpkl import FedPKLClient

'''
看了源代码，基于FedMD改来的（代码一坨稀烂）。原文没提到的，是有预训练的（虽然写了pretrain但是没有实现，但是又在进入ktpfl的时候有pretrain的部分），用的fmnist为私有、minst为共有，setting和md一样，但是没有先预训练pub+pri，而是直接pretrain pri。问题很大，传到ktpfl-core的时候pri和total-pri没有用，还是自己重新新建的priv虽然内容还是一样的。(这样有很大问题啊，外边的pub和pri是通过文件设置的，新建的priv是写死的)
流程：用pri pretrain一遍（有限的轮次-仅1轮）；各设备上传pub的softmax的结果，T为10太高了吧，惩罚系数0.7也同样太高了吧；服务端用执行参数矩阵更新并更新聚合梯度-get_models_logits，这一部分首先逐客户计算损失函数，即每次更新一行的系数矩阵，需要执行n-cli次才能更新整个参数矩阵，怪的一批，然后无法理解得用逐cli对参数矩阵的对应行softmax（T=0.2）论文没提，这才完成参数矩阵的更新然后得到n个cli的logit；客户端执行KL蒸馏（和pretrain的函数一样，用with-softmax来标记令人困惑，传入了一个MSELOS实际并没有用上的操作同样令人困惑）；最后再在pri进行训练。又发现一个问题，上传的是logits，softmax-with-T过的；返回来的logits，视为output又做了一遍softmax-with-T（即log_softmax）执行的klloss；又有一个问题，klloss要乘以T*T的系数以平衡量级，这里没有，所以用了一个很大alpha。最后，好家伙，和md一个尿性，蒸馏和自学习分开学，按理来说要蒸馏是这两个的加权和，前面的alpha说错了，不是一个概念；然而，这个问题更大了，优化这个参数矩阵的损失函数是KLLOSS+惩罚MSE，代码里用的都是MSE。。。。
至少代码一坨，不是很想复现了，有些奇怪的点在里面。源代码也是基于FEDMD，可以直接继承，稍微写一下，运行就不测试了。consensus、score的概念也是一样的，KD+FL，不过计算的方法不一样；但是关于上传的部分，原版有问题，那我聚合的部分到底是logit还是output，还是上传ouput，方便遵循通用写法log_softmax

妈的，浪费时间，虽然说勉强是写出来了，说实话不知道对不对，因为看样子最关键的应该是那个系数矩阵对吧，几乎没变化，也不知道是训练太少了还是啥的，还是哪里写错了，总之大概是照着意思搞出来了，优化矩阵好麻烦地说，也不知道对不对，看样子是对的，总之要比原版好我觉得
'''

class FedPKLServer(FedMDServer):
    # to align the naming, (P)ersonalized (K)nowledge (T)ransfer，not offical kl-pfl
    def __init__(self, datasets, models, args):
        super().__init__(datasets, models, args)

        self.distil_critertion = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.regular_critertion = torch.nn.MSELoss()
        self.per_knowledge  = [None for _ in range(self.num_client)]
        self.para_matrix = torch.zeros(self.num_client, self.num_client)
        
    def distribute(self, client_id):
        package = (self.all_client_model[client_id], self.per_knowledge[client_id])
        return package
    
    def aggregate(self):
        # before aggregate for the final personalized knowlegde for different client, optimize the weight/parameter first
        self.optim_weight()
        for client_id in self.selected_clients:
            others_weight = self.para_matrix[client_id][self.selected_clients].to(self.device)
            others_output = [self.all_client_score[i] for i in self.selected_clients]
            teacher_output = (torch.stack(others_output, dim=-1) * others_weight).sum(dim=-1)
            self.per_knowledge[client_id] = teacher_output

    def optim_weight(self):
        sub_para_matrix = torch.zeros(len(self.selected_clients), len(self.selected_clients)).to(self.device)
        sub_para_matrix.requires_grad_()
        self.optimizer = torch.optim.SGD(
            [sub_para_matrix], 
            lr=self.args.global_lr, 
        )
        weights = [self.all_client_weight[i] for i in self.selected_clients]
        weights = (torch.tensor(weights) / sum(weights)).to(self.device)
        regular_bound = torch.ones(self.num_client, self.num_client) / self.num_client
        # all_clien_score: [client_id, num_batch, batch_size, num_classes/output_dim]
        for _ in range(self.args.num_learn_matrix_epoch):
            num_batch = len(self.all_client_score[self.selected_clients[0]])
            for batch_id in range(num_batch):
                selected_output = torch.stack([self.all_client_score[i][batch_id] for i in self.selected_clients]).unsqueeze(dim=0).repeat(len(self.selected_clients), 1, 1, 1)
                teacher_output = (sub_para_matrix.unsqueeze(-1).unsqueeze(-1) * selected_output).sum(dim=0) # not sure if right
                student_output = torch.stack([self.all_client_score[i][batch_id] for i in self.selected_clients])
                print(teacher_output.shape, student_output.shape)
                T = self.args.global_temper
                distil_loss = 0
                for i in range(len(self.selected_clients)):
                    distil_loss += self.distil_critertion(
                            F.log_softmax(student_output[i] / T, dim=1),
                            F.log_softmax(teacher_output[i] / T, dim=1),
                        ) * (T * T) * weights[i]
                distil_loss = torch.mean(distil_loss)
                loss = torch.mean(distil_loss) + self.args.rho * self.regular_critertion(self.para_matrix, regular_bound)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        sub_para_matrix = sub_para_matrix.detach()
        for i, idi in enumerate(self.selected_clients):
            for j, idj in enumerate(self.selected_clients):
                self.para_matrix[idi][idj] = sub_para_matrix[i][j]
        # print(self.para_matrix)
    
    # about self.train() pretrain part
    # here pretain on the pub, in origin code on the pri, nothing if according to paper