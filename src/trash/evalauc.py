def eval(self):
    self.model.eval()
    num_samples,ls, acc = 0, 0, 0
    preds, trues = [], []
    # torch.no_grad() 上下文管理器是用于关闭梯度跟踪的，而 detach() 方法是用于分离张量的。在推断阶段，使用 torch.no_grad() 可以关闭梯度跟踪，而不需要使用 detach() 方法。在训练阶段，如果需要保留梯度信息，可以使用 detach() 方法来分离张量。
    with torch.no_grad():
        for X, y in (self.testloader):
            X, y = X.to(self.device), y.to(self.device)
            output = self.model(X)
            loss = self.criterion(output, y)
            
            num_samples += y.size(0)
            ls += loss.item() * num_samples # reduction is mean by default
            acc += (torch.argmax(output.detach(), dim=1) == y).sum()
            
            # preds.append(output.detach().cpu().numpy())
            # trues.append(F.one_hot(y.to(torch.int64), num_classes=len(self.testset.classes)).detach().cpu().numpy()) 1. with torch.nn.F
            # trues.append(label_binarize((y.detach().cpu().numpy()), classes=np.arange(len(self.dataset.num_class)))) # 2. not useful, the classes part

    # preds, trues = np.concatenate(preds), np.concatenate(trues)
    # auc = metrics.roc_auc_score(trues, preds, average='macro')
    acc = acc / num_samples * 100
    ls = ls / num_samples
    print(f'client{self.id}, test loss:{ls:.3f}, test accuracy:{acc:.2f}%')
    
    return ls, acc

# 就是注释的几段，主要是感觉没必要啊，转换成one-hot向量，折腾的好一会，妈的后面遇到特殊情况（不想改也没必要）