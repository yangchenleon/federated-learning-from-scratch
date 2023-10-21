import torch
import torch.nn as nn
from data.utils.setting import data_info

class MLP(nn.Module):
    def __init__(self, dataset):
        super(MLP, self).__init__()
        input_dim = data_info[dataset][0] * data_info[dataset][1]
        output_dim = data_info[dataset][2]
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.init_weights(self)  # 在构造函数中调用权重初始化方法
    
    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = self.flat(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    @staticmethod
    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                # nn.init.xavier_normal_(layer.weight.data)

class AlexNet(nn.Module):
    def __init__(self, dataset):
        super(AlexNet, self).__init__()
        input_dim = data_info[dataset][0]
        output_dim = data_info[dataset][2]
        self.features = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(4096, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.fc(x)
        return x

ModelDict = {
    'mlp': MLP,
    'alexnet': AlexNet
}