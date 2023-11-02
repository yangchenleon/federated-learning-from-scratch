import torch
import torch.nn.functional as F

from src.client.fedmd import FedMDClient


class FedPKLClient(FedMDClient):
    # to align the naming, (P)ersonalized (K)nowledge (T)ransfer，not offical kl-pfl
    def __init__(self, client_id, dataset, model, args, logger, device):
        super().__init__(client_id, dataset, model, args, logger, device)
        
        # for simplicity, keep the consensus, use as the personalized knowledge
        # also for simplicity, upload the ouput not tempered softmax result (want keep the scheme using log_softmax to do klloss, not use only the log[don't know how actually])
        self.distil_critertion = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)# to learning personalized logits, don't know why differ mean and batchmean, look same 
  
    def distill(self):
        # the only difference between digest() is the use of kl loss
        self.model.train()
        num_samples, ls = 0, 0
        digest_ls = []
        for _ in range(self.args.num_distil_epoch):
            for i, data in enumerate(self.publicloader):
                X, y = data[0].to(self.device), data[1].to(self.device)
                output = self.model(X)
                loss = self.distil_critertion(
                    F.log_softmax(output / self.args.local_temper), 
                    F.log_softmax(self.consensus[i] / self.args.local_temper),
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_samples += y.size(0)
                ls += loss.item() * num_samples
            
            digest_ls.append(ls/num_samples)
        return digest_ls
    
    def train(self, save=True):
        # switch the train order
        train_ls, train_acc = super().train(save)
        digest_ls = self.distill()
        return digest_ls, train_ls, train_acc