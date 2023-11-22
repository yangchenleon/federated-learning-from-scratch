import re

from src.models.simple import MLP, CNN, CNN2L, CNN3L, CNN4L, LeNet 
from src.models.offical import AlexNet, ResNet, MobileNet
from src.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from src.models.efficientnet import EfficientNetB0
from src.models.shufflenet import ShuffleNetG2, ShuffleNetG3
from src.models.mobilenet import MobileNetV2
from src.models.alexnet import TheAlexNet
from src.models.dla import DLA


ModelDict = {
    # self-defined simple model
    'mlp': MLP,
    'cnn': CNN,
    'cnn2': CNN2L,
    'cnn3': CNN3L,
    'cnn4': CNN4L,
    'lenet': LeNet,
    # below offical model
    'alexnet': AlexNet,
    'resnet': ResNet,
    'mobile': MobileNet,
    # below only suit for 10-class tasks
    'theres18': ResNet18,
    'theres34': ResNet34,
    'theres50': ResNet50,
    'theres101': ResNet101,
    'theres152': ResNet152,
    'theeffi0': EfficientNetB0,
    'theshuf2': ShuffleNetG2,
    'theshuf3': ShuffleNetG3,
    'themobi2': MobileNetV2,
    'thealex': TheAlexNet,
    'thedla': DLA,
}

def getmodel(model_name, num_classes=10):
    if model_name[:3] == 'the': # seems only work on 10-classes task(mnist, cifar10..), can try to modify, whatever
        model = ModelDict[model_name]() 
    else: # official model with classifier and base
        pattern = r'([A-Za-z]+)(\d+)'
        match = re.match(pattern, model_name)
        if match is not None:
            family, version = match.group(1), match.group(2)
        else:
            family, version = model_name, 0 # a path for alexnet(only one has no varient)
        model = ModelDict[family](version, num_classes)
    
    return model