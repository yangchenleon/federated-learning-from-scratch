from src.models.simple import MLP, CNN
from src.models.offical import AlexNet, ResNet18, MobileNetV2
from src.models.resnet import TheResNet18
from src.models.alexnet import TheAlexNet


ModelDict = {
    'mlp': MLP,
    'cnn': CNN,
    'alexnet': AlexNet,
    'resnet18': ResNet18,
    'mobile': MobileNetV2,
    'theres18': TheResNet18,
    'thealex': TheAlexNet
}