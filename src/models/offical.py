import os
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from data.utils.setting import data_info

os.environ['TORCH_HOME']='results/pretrained/'


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