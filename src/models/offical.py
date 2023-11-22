import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M

os.environ['TORCH_HOME']='results/pretrained/'


class AlexNet(nn.Module):
    def __init__(self, version, num_classes=10):
        super(AlexNet, self).__init__()
        pretrained = False
        self.base = M.alexnet(
            weights=M.AlexNet_Weights.DEFAULT if pretrained else None
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

class ResNet(nn.Module):
    def __init__(self, version, num_classes=10):
        super().__init__()
        archs = {
            "18": (M.resnet18,M.ResNet18_Weights.DEFAULT),
            "34": (M.resnet34, M.ResNet34_Weights.DEFAULT),
            "50": (M.resnet50, M.ResNet50_Weights.DEFAULT),
            "101": (M.resnet101, M.ResNet101_Weights.DEFAULT),
            "152": (M.resnet152, M.ResNet152_Weights.DEFAULT),
        }

        pretrained = False
        resnet: M.ResNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = resnet
        self.classifier = nn.Linear(self.base.fc.in_features, num_classes)
        self.base.fc = nn.Identity()
    
    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x
    
class SqueezeNet(nn.Module):
    def __init__(self, version, num_classes=10):
        super().__init__()

        pretrained = False
        archs = {
            "0": (M.squeezenet1_0, M.SqueezeNet1_0_Weights.DEFAULT),
            "1": (M.squeezenet1_1, M.SqueezeNet1_1_Weights.DEFAULT),
        }
        squeezenet: M.SqueezeNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = squeezenet.features
        self.classifier = nn.Sequential( # see the source code, will know why
            nn.Dropout(),
            nn.Conv2d(squeezenet.classifier[1].in_channels, num_classes, kernel_size=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

class DenseNet(nn.Module):
    def __init__(self, version, num_classes=10):
        super().__init__()
        
        archs = {
            "121": (M.densenet121, M.DenseNet121_Weights.DEFAULT),
            "161": (M.densenet161, M.DenseNet161_Weights.DEFAULT),
            "169": (M.densenet169, M.DenseNet169_Weights.DEFAULT),
            "201": (M.densenet201, M.DenseNet201_Weights.DEFAULT),
        }
        pretrained = False
        densenet: M.DenseNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = densenet
        self.classifier = nn.Linear(
            densenet.classifier.in_features, num_classes
        )
        self.base.classifier = nn.Identity()

class MobileNet(nn.Module):
    def __init__(self, version, num_classes=10):
        super().__init__()
        
        archs = {
            "2": (M.mobilenet_v2, M.MobileNet_V2_Weights.DEFAULT),
            "30": (M.mobilenet_v3_small, M.MobileNet_V3_Small_Weights.DEFAULT),
            "31": (M.mobilenet_v3_large, M.MobileNet_V3_Large_Weights.DEFAULT),
        }

        pretrained = False
        mobilenet = archs[version][0](weights=archs[version][1] if pretrained else None)
        self.base = mobilenet
        self.classifier = nn.Linear(
            mobilenet.classifier[-1].in_features, num_classes
        )
        self.base.classifier[-1] = nn.Identity()

class EfficientNet(nn.Module):
    def __init__(self, version, num_classes=10):
        super().__init__()
        
        archs = {
            "0": (M.efficientnet_b0, M.EfficientNet_B0_Weights.DEFAULT),
            "1": (M.efficientnet_b1, M.EfficientNet_B1_Weights.DEFAULT),
            "2": (M.efficientnet_b2, M.EfficientNet_B2_Weights.DEFAULT),
            "3": (M.efficientnet_b3, M.EfficientNet_B3_Weights.DEFAULT),
            "4": (M.efficientnet_b4, M.EfficientNet_B4_Weights.DEFAULT),
            "5": (M.efficientnet_b5, M.EfficientNet_B5_Weights.DEFAULT),
            "6": (M.efficientnet_b6, M.EfficientNet_B6_Weights.DEFAULT),
            "7": (M.efficientnet_b7, M.EfficientNet_B7_Weights.DEFAULT),
        }

        pretrained = False
        efficientnet: M.EfficientNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = efficientnet
        self.classifier = nn.Linear(
            efficientnet.classifier[-1].in_features, num_classes
        )
        self.base.classifier[-1] = nn.Identity()