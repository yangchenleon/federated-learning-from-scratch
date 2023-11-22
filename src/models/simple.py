import torch.nn as nn
import torch.nn.functional as F
'''
随便找个地方记录算了：nn.Flatten(x) == x.view(x.size(0), -1)
'''

class MLP(nn.Module):
    def __init__(self, version, num_class=10):
        super(MLP, self).__init__()
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_class)
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
 
class CNN(nn.Module):
    def __init__(self, version, num_class=10):
        super(CNN, self).__init__() # channel dim
        self.base = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            # nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 512),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, num_class)
        )
    
    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x

class CNN2L(nn.Module):
    def __init__(self, version, num_class=10):
        super(CNN2L, self).__init__() # channel dim
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), 
            nn.ReLU(),
            nn.Linear(128, num_class),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x

class CNN3L(nn.Module):
    def __init__(self, version, num_class=10):
        super(CNN3L, self).__init__() # channel dim
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 * 3, 256), # can add a AdaptiveAvgPool2d before
            nn.ReLU(),
            nn.Linear(256, num_class),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x

class CNN4L(nn.Module):
    def __init__(self, version, num_class=10):
        super(CNN4L, self).__init__() # channel dim
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, num_class),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x

class LeNet(nn.Module):
    def __init__(self, version, num_class=10):
        super(LeNet, self).__init__()        
        self.base = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_class)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x