import os
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

os.environ['TORCH_HOME']='results/pretrained/'


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        pretrained = False
        self.base = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.DEFAULT if pretrained else None
        )
        self.classifier = nn.Linear(
            self.base.classifier[-1].in_features, num_classes
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
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        pretrained = False
        self.base = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.classifier = nn.Linear(self.base.fc.in_features, num_classes)
        self.base.fc = nn.Identity()
    
    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x
    
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        pretrained = False
        self.base = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.classifier = nn.Linear(self.base.classifier[-1].in_features, num_classes)

        self.base.classifier[-1] = nn.Identity()

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x