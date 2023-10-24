

# 上传聚合的一种方法（可以，比较拧巴，写法略显花里胡哨，总之就是把模型拆成字典，也以拆分形式传输，最后通过）
def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module],
    detach=False,
    requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:
    """Collect all parameters in `src` that `.requires_grad = True` into a list and return it.

    Args:
        src (Union[OrderedDict[str, torch.Tensor], torch.nn.Module]): The source that contains parameters.
        requires_name (bool, optional): If set to `True`, The names of parameters would also return in another list. Defaults to False.
        detach (bool, optional): If set to `True`, the list would contain `param.detach().clone()` rather than `param`. Defaults to False.

    Returns:
        Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]: List of parameters, [List of names of parameters].
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)

    if requires_name:
        return parameters, keys
    else:
        return parameters
random_init_params, self.trainable_params_name = trainable_params(
    self.model, detach=True, requires_name=True
)
self.global_params_dict = OrderedDict(
    zip(self.trainable_params_name, random_init_params)
)
import torch
def aggregate(self):
    weights = torch.tensor(self.client_weight, device=self.device) / sum(self.client_weight)
    for glo_layer, cli_layer in zip(self.global_model.parameters(), zip(*self.client_model)):
        glo_layer.data = (torch.stack(cli_layer, dim=-1) * weights).sum(dim=-1)
    self.global_model.load_state_dict(self.global_model, strict=False)

# 搭配
def upload(self):
    # 2. with state_dict()
    parameters, keys = [], [] # keys is acutally useless
    for name, param in self.model.state_dict(keep_vars=True).items(): # in case need gradient
        if param.requires_grad:
            parameters.append(param.detach().clone()) # .data equal to .detach()
            keys.append(name)
    return parameters, len(self.trainset)

# 上传聚合的第二种方法（不行啊，看着挺直观，但模型就是没有聚合的效果，传输的就是一整个模型，通过parameter层直接修改数据，感觉是涉及深浅拷贝的问题）
def aggregate2(self):
    for param in self.global_model.parameters():
        param.data.zero_()
    w = torch.tensor(self.client_weight) / sum(self.client_weight)
    print(w)
    for w, model in zip(w, self.client_model):
        self.add_parameters(w, model)

def add_parameters(self, w, model):
    for server_param, client_param in zip(self.global_model.parameters(), model.parameters()):
        server_param.data += client_param.data.clone() * w

# 客户端（实测这种方法，理论上全局模型在本地性能会下降但不是没有，这里经常出现0.0，1.24且在下一次同一个客户端还是这昂）
def receive(self, model):
    for new_param, old_param in zip(model.parameters(), self.model.parameters()):
        old_param.data = new_param.data.clone()