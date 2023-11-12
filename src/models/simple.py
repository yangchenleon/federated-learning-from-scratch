import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_class=10):
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
    def __init__(self, num_class=10):
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