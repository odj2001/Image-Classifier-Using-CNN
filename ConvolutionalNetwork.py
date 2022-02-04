import torch.nn as nn
import torch.nn.functional as f
from torch.nn import Softmax


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)

        self.layer5 = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(57600, 5)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        sm = nn.Softmax(1)
        return sm(x)
