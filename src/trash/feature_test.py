class MyClass:
    def __init__(self, name):
        self.dataset = name
        # self.para1 = None

    def my_method(self, para1):
        dataset = "Method dataset"
        self.para1 = para1
        # print(dataset)  # 输出: Method dataset

        # # 使用类变量
        # print(self.dataset)  # 输出: Original dataset
    def printextra(self):
        print(self.para1)

class SonClass(MyClass):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age
    def fuck(self):
        dataset = self.dataset + "fdsfds"
        print(self.dataset)
        print(dataset)
# 创建类实例

obj = SonClass('666', 2323)

# 调用方法
# obj.fuck()

obj = MyClass('suck')
# obj.my_method('duck')
# obj.printextra()

# ------------------------------------
import torch
import numpy as np
from torchvision import transforms
a = np.array([[[
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   1.,   0.,   0.,  13.,  73.,   0.,   0.,   1.,   4.,   0.,
    0.,   0.,   0.,   1.,   1.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   3.,   0.,  36., 136., 127.,  62.,  54.,   0.,   0.,   0.,
    1.,   3.,   4.,   0.,   0.,   3.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   6.,   0., 102., 204., 176., 134., 144., 123.,  23.,   0.,
    0.,   0.,   0.,  12.,  10.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0., 155., 236., 207., 178., 107., 156., 161., 109.,
    64.,  23.,  77., 130.,  72.,  15.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    1.,   0.,  69., 207., 223., 218., 216., 216., 163., 127., 121.,
    122., 146., 141.,  88., 172.,  66.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   1.,
    1.,   0., 200., 232., 232., 233., 229., 223., 223., 215., 213.,
    164., 127., 123., 196., 229.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0., 183., 225., 216., 223., 228., 235., 227., 224., 222.,
    224., 221., 223., 245., 173.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0., 193., 228., 218., 213., 198., 180., 212., 210., 211.,
    213., 223., 220., 243., 202.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   3.,
    0.,  12., 219., 220., 212., 218., 192., 169., 227., 208., 218.,
    224., 212., 226., 197., 209.,  52.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   6.,
    0.,  99., 244., 222., 220., 218., 203., 198., 221., 215., 213.,
    222., 220., 245., 119., 167.,  56.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   4.,   0.,
    0.,  55., 236., 228., 230., 228., 240., 232., 213., 218., 223.,
    234., 217., 217., 209.,  92.,   0.],
    [  0.,   0.,   1.,   4.,   6.,   7.,   2.,   0.,   0.,   0.,   0.,
    0., 237., 226., 217., 223., 222., 219., 222., 221., 216., 223.,
    229., 215., 218., 255.,  77.,   0.],
    [  0.,   3.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  62., 145.,
    204., 228., 207., 213., 221., 218., 208., 211., 218., 224., 223.,
    219., 215., 224., 244., 159.,   0.],
    [  0.,   0.,   0.,   0.,  18.,  44.,  82., 107., 189., 228., 220.,
    222., 217., 226., 200., 205., 211., 230., 224., 234., 176., 188.,
    250., 248., 233., 238., 215.,   0.],
    [  0.,  57., 187., 208., 224., 221., 224., 208., 204., 214., 208.,
    209., 200., 159., 245., 193., 206., 223., 255., 255., 221., 234.,
    221., 211., 220., 232., 246.,   0.],
    [  3., 202., 228., 224., 221., 211., 211., 214., 205., 205., 205.,
    220., 240.,  80., 150., 255., 229., 221., 188., 154., 191., 210.,
    204., 209., 222., 228., 225.,   0.],
    [ 98., 233., 198., 210., 222., 229., 229., 234., 249., 220., 194.,
    215., 217., 241.,  65.,  73., 106., 117., 168., 219., 221., 215.,
    217., 223., 223., 224., 229.,  29.],
    [ 75., 204., 212., 204., 193., 205., 211., 225., 216., 185., 197.,
    206., 198., 213., 240., 195., 227., 245., 239., 223., 218., 212.,
    209., 222., 220., 221., 230.,  67.],
    [ 48., 203., 183., 194., 213., 197., 185., 190., 194., 192., 202.,
    214., 219., 221., 220., 236., 225., 216., 199., 206., 186., 181.,
    177., 172., 181., 205., 206., 115.],
    [  0., 122., 219., 193., 179., 171., 183., 196., 204., 210., 213.,
    207., 211., 210., 200., 196., 194., 191., 195., 191., 198., 192.,
    176., 156., 167., 177., 210.,  92.],
    [  0.,   0.,  74., 189., 212., 191., 175., 172., 175., 181., 185.,
    188., 189., 188., 193., 198., 204., 209., 210., 210., 211., 188.,
    188., 194., 192., 216., 170.,   0.],
    [  2.,   0.,   0.,   0.,  66., 200., 222., 237., 239., 242., 246.,
    243., 244., 221., 220., 193., 191., 179., 182., 182., 181., 176.,
    166., 168.,  99.,  58.,   0.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,  40.,  61.,  44.,  72.,
    41.,  35.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.]]]
    ,[[
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   1.,   0.,   0.,  13.,  73.,   0.,   0.,   1.,   4.,   0.,
    0.,   0.,   0.,   1.,   1.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   3.,   0.,  36., 136., 127.,  62.,  54.,   0.,   0.,   0.,
    1.,   3.,   4.,   0.,   0.,   3.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   6.,   0., 102., 204., 176., 134., 144., 123.,  23.,   0.,
    0.,   0.,   0.,  12.,  10.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0., 155., 236., 207., 178., 107., 156., 161., 109.,
    64.,  23.,  77., 130.,  72.,  15.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    1.,   0.,  69., 207., 223., 218., 216., 216., 163., 127., 121.,
    122., 146., 141.,  88., 172.,  66.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   1.,
    1.,   0., 200., 232., 232., 233., 229., 223., 223., 215., 213.,
    164., 127., 123., 196., 229.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0., 183., 225., 216., 223., 228., 235., 227., 224., 222.,
    224., 221., 223., 245., 173.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0., 193., 228., 218., 213., 198., 180., 212., 210., 211.,
    213., 223., 220., 243., 202.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   3.,
    0.,  12., 219., 220., 212., 218., 192., 169., 227., 208., 218.,
    224., 212., 226., 197., 209.,  52.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   6.,
    0.,  99., 244., 222., 220., 218., 203., 198., 221., 215., 213.,
    222., 220., 245., 119., 167.,  56.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   4.,   0.,
    0.,  55., 236., 228., 230., 228., 240., 232., 213., 218., 223.,
    234., 217., 217., 209.,  92.,   0.],
    [  0.,   0.,   1.,   4.,   6.,   7.,   2.,   0.,   0.,   0.,   0.,
    0., 237., 226., 217., 223., 222., 219., 222., 221., 216., 223.,
    229., 215., 218., 255.,  77.,   0.],
    [  0.,   3.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  62., 145.,
    204., 228., 207., 213., 221., 218., 208., 211., 218., 224., 223.,
    219., 215., 224., 244., 159.,   0.],
    [  0.,   0.,   0.,   0.,  18.,  44.,  82., 107., 189., 228., 220.,
    222., 217., 226., 200., 205., 211., 230., 224., 234., 176., 188.,
    250., 248., 233., 238., 215.,   0.],
    [  0.,  57., 187., 208., 224., 221., 224., 208., 204., 214., 208.,
    209., 200., 159., 245., 193., 206., 223., 255., 255., 221., 234.,
    221., 211., 220., 232., 246.,   0.],
    [  3., 202., 228., 224., 221., 211., 211., 214., 205., 205., 205.,
    220., 240.,  80., 150., 255., 229., 221., 188., 154., 191., 210.,
    204., 209., 222., 228., 225.,   0.],
    [ 98., 233., 198., 210., 222., 229., 229., 234., 249., 220., 194.,
    215., 217., 241.,  65.,  73., 106., 117., 168., 219., 221., 215.,
    217., 223., 223., 224., 229.,  29.],
    [ 75., 204., 212., 204., 193., 205., 211., 225., 216., 185., 197.,
    206., 198., 213., 240., 195., 227., 245., 239., 223., 218., 212.,
    209., 222., 220., 221., 230.,  67.],
    [ 48., 203., 183., 194., 213., 197., 185., 190., 194., 192., 202.,
    214., 219., 221., 220., 236., 225., 216., 199., 206., 186., 181.,
    177., 172., 181., 205., 206., 115.],
    [  0., 122., 219., 193., 179., 171., 183., 196., 204., 210., 213.,
    207., 211., 210., 200., 196., 194., 191., 195., 191., 198., 192.,
    176., 156., 167., 177., 210.,  92.],
    [  0.,   0.,  74., 189., 212., 191., 175., 172., 175., 181., 185.,
    188., 189., 188., 193., 198., 204., 209., 210., 210., 211., 188.,
    188., 194., 192., 216., 170.,   0.],
    [  2.,   0.,   0.,   0.,  66., 200., 222., 237., 239., 242., 246.,
    243., 244., 221., 220., 193., 191., 179., 182., 182., 181., 176.,
    166., 168.,  99.,  58.,   0.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,  40.,  61.,  44.,  72.,
    41.,  35.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.],
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
    0.,   0.,   0.,   0.,   0.,   0.]]]], dtype=np.uint8)
# print(a.dtype, a.shape)
# b = transforms.ToTensor()(a)
# print(b.dtype, b.shape)

# ------------------------------------
# import torchvision
# testset = torchvision.datasets.CIFAR10('data/datasets/CIFAR10/raw', train=False, download=True)
# print(torch.Tensor(testset.data).shape)

# ------------------------------------
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
# y_true = np.array([0, 0, 1, 1])
# y_predprob =np.array([[0.9,0.1],[0.6,0.4],[0.65,0.35],[0.2,0.8]])
# trues, preds = [], []

# trues.append(F.one_hot(torch.tensor(y_true, dtype=torch.int64), num_classes=2).numpy())
# preds.append(y_predprob)
# print(preds, trues)
# # preds, trues = np.concatenate(preds), np.concatenate(trues)
# # print(preds, trues)
# auc = roc_auc_score(trues, preds, average='micro')
# print(auc)#0.75

# ------------------------------------
# print(np.concatenate([np.array([[0.9 , 0.1 ],
#        [0.6 , 0.4 ],
#        [0.65, 0.35],
#        [0.2 , 0.8 ]]),
#        np.array([[0.9 , 0.1 ],
#        [0.6 , 0.4 ],
#        [0.65, 0.35],
#        [0.2 , 0.8 ]])] ,axis=0))

# ------------------------------------
# import random
# print([random.sample(range(10), 2) for _ in range(5)])

# ------------------------------------
import torch.nn as nn
import torchvision
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.base = torchvision.models.alexnet(
            weights=None
        )
        self.classifier = nn.Linear(
            self.base.classifier[-1].in_features, 10
        )
        self.base.classifier[-1] = nn.Identity()

        # self.model = torchvision.models.alexnet(
        #     weights=torchvision.models.AlexNet_Weights.DEFAULT if pretrained else None,
        #     num_classes = output_dim
        # )

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x
    
model = AlexNet()
# print(isinstance(model, nn.Module))
# print(len(list(model.parameters())), len(list(model.modules())))
# for i in model.parameters():
#     print(i.shape,)# i.data
# for i in model.modules():
#     print(i)
# for name, tensor in model.state_dict(keep_vars=True).items():
#     print(name, tensor.detach().clone())

# ------------------------------------
import random
a = random.choices([1], k=5)
# print(a)

# ------------------------------------
a = {'yi': 1, 'er':2, 'san':3}
b = [3,2,1]
# for  val, num in zip(a, b):
    # print( val, num)

# ------------------------------------
import logging

# # 创建logger实例
# logger = logging.getLogger('Federated Learning')
# logger.setLevel(logging.INFO)

# # 创建文件处理器
# file_handler = logging.FileHandler('federated_learning.log')

# # 设置日志格式
# formatter = logging.Formatter('%(relativeCreated)s - %(message)s', '%H:%M:%S')
# file_handler.setFormatter(formatter)

# # 将文件处理器添加到logger
# logger.addHandler(file_handler)

# # 在代码中使用logger记录日志
# logger.info('Federated learning process started.')
# logger.debug('Debug information.')
# logger.warning('Warning message.')
# logger.error('Error message.')

# # 移除处理器和关闭文件处理器
# logger.removeHandler(file_handler)
# file_handler.close()

# ----------------------------------- 
# from argparse import ArgumentParser

# # 创建两个父级解析器，禁用帮助选项
# parent_parser1 = ArgumentParser(add_help=False)
# parent_parser2 = ArgumentParser(add_help=False)

# # 向 parent_parser1 添加参数
# parent_parser1.add_argument('--arg1', type=int, help='Argument 1')

# # 向 parent_parser2 添加参数
# parent_parser2.add_argument('--arg2', type=str, help='Argument 2')

# # 创建子解析器，并指定父级解析器
# child_parser = ArgumentParser(parents=[parent_parser1, parent_parser2])

# # 添加子解析器自己的参数
# child_parser.add_argument('--arg3', type=float, help='Argument 3')

# # 使用 parse_args() 解析命令行参数
# args = child_parser.parse_args()

# # 访问解析结果
# print(args.arg1)
# print(args.arg2)
# print(args.arg3)

# ----------------------------------------
import torch
import torch.optim as optim

# # 初始化权重和模型
# modela = torch.tensor([1.0, 2.0, 3.0])
# w = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
# modelb = torch.tensor([1.0, 2.0, 3.0])

# learning_rate = 0.01
# optimizer = optim.SGD([w], lr=learning_rate)

# for i in range(100):
#     model_output = modela +  w * modelb
#     loss = torch.abs(torch.sum(model_output))
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(model_output)
# with torch.no_grad():
#     print(w.detach().tolist())

# --------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# # 生成两个随机变量的数据
# x = np.random.randn(1000)
# y = np.random.randn(1000)

# # 绘制二元直方图
# plt.hist2d(x, y, bins=20, cmap='Blues')

# # 添加颜色条
# plt.colorbar()

# # 设置坐标轴标签和标题
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('2D Histogram')

# # 显示图像
# plt.savefig(f'results/figures/test.png')

# -------------------------------------------
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

# 假设有n个人
n = 10

# 假设进行100次筛子
num_trials = 100

# 生成筛子结果数据
dice_results = np.random.randint(1, 7, size=(num_trials, n))

# 统计每个点数的次数
counts = np.zeros((6, n))
for i in range(6):
    for j in range(n):
        counts[i, j] = np.sum(dice_results[:, j] == (i + 1))

# 创建画布和轴
fig, ax = plt.subplots()

# 调整数据形状以匹配散点图要求
x = np.repeat(np.arange(n), 6)
y = np.tile(np.arange(1, 7), n)
sizes = counts.flatten()
colors = counts.flatten()

# 绘制散点图
ax.scatter(x, y, c=colors, s=sizes, cmap='Blues')

# 添加网格
ax.grid(True, linestyle='--', linewidth=0.5)

# 设置图表标题和轴标签
ax.set_title('Distributed Dot Plot - Dice Results')
ax.set_xlabel('Person')
ax.set_ylabel('Point')

# 设置横轴刻度范围和标签
ax.set_xlim(-1, n)
ax.set_xticks(np.arange(n))
ax.set_xticklabels(np.arange(1, n+1))

# 设置纵轴刻度范围和标签
ax.set_ylim(0, 7)
ax.set_yticks(np.arange(1, 7))
ax.set_yticklabels(np.arange(1, 7))

# 显示图表
plt.show()

# 显示图像
plt.savefig(f'results/figures/test.png')

# --------------------------------

import matplotlib.pyplot as plt
import numpy as np

# # 模拟投掷骰子的结果
# def simulate_dice_rolls(num_users, num_rolls):
#     rolls = np.random.randint(1, 7, size=(num_users, num_rolls))
#     return rolls

# # 统计每个用户每个点数的次数
# def count_dice_rolls(rolls):
#     counts = np.zeros((rolls.shape[0], 6), dtype=int)
#     for i in range(6):
#         counts[:, i] = np.sum(rolls == (i+1), axis=1)
#     return counts

# # 生成分布直方图
# def plot_histogram(counts):
#     num_users = counts.shape[0]
#     x = np.arange(num_users)
    
#     fig, ax = plt.subplots()
#     bar_width = 0.5

#     colors = ['r', 'g', 'b', 'c', 'm', 'y']
#     labels = ['1', '2', '3', '4', '5', '6']
    
#     for i in range(6):
#         ax.bar(x, counts[:, i], bar_width, label=labels[i], color=colors[i], bottom=np.sum(counts[:, :i], axis=1))

#     ax.set_xlabel('User')
#     ax.set_ylabel('Count')
#     ax.set_title('Dice Roll Distribution')
#     ax.set_xticks(x)
#     ax.set_xticklabels([str(i+1) for i in range(num_users)])
#     ax.legend()

#     plt.tight_layout()
#     plt.show()

# # 主函数
# def main():
#     num_users = 10
#     num_rolls = 100

#     rolls = simulate_dice_rolls(num_users, num_rolls)
#     counts = count_dice_rolls(rolls)
#     plot_histogram(counts)

# if __name__ == '__main__':
#     main()

# plt.savefig(f'results/figures/Distribution of Dice Rolls11.png')