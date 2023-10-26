import os
os.environ['TORCH_HOME']='results/pretrained/'
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from data.utils.setting import data_info


class MLP(nn.Module):
    def __init__(self, dataset, pretrained):
        super(MLP, self).__init__()
        pretrained = False
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

class PerAlexNet(nn.Module):
    def __init__(self, dataset, pretrained):
        super(PerAlexNet, self).__init__()
        input_dim = data_info[dataset][0]
        output_dim = data_info[dataset][2]
        pretrained = False
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
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

class AlexNet(nn.Module):
    def __init__(self, dataset, pretrained=False):
        input_dim = data_info[dataset][0] * data_info[dataset][1]
        output_dim = data_info[dataset][2]
        super(AlexNet, self).__init__()
        self.base = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.DEFAULT if pretrained else None
        )
        self.classifier = nn.Linear(
            self.base.classifier[-1].in_features, output_dim
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

class ResNet18(nn.Module):
    def __init__(self, dataset, pretrained=False):
        super(ResNet18, self).__init__()
        output_dim = data_info[dataset][2]
        self.base = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.classifier = nn.Linear(self.base.fc.in_features, output_dim)
        self.base.fc = nn.Identity()
    
    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x
    
class MobileNetV2(nn.Module):
    def __init__(self, dataset, pretrained=False):
        output_dim = data_info[dataset][2]
        super(MobileNetV2, self).__init__()
        self.base = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.classifier = nn.Linear(self.base.classifier[-1].in_features, output_dim)

        self.base.classifier[-1] = nn.Identity()

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x

ModelDict = {
    'mlp': MLP,
    'alexnet': AlexNet,
    'resnet18': ResNet18,
    'peralex': PerAlexNet,
    'mobile': MobileNetV2
}