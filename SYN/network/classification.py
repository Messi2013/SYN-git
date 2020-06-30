import torch.nn as nn
import torch
import torch.nn.functional as F


class Classification(nn.Module):
    def __init__(self, in_size=None, in_channels=None, num_classes=None):
        super(Classification, self).__init__()
        in_height, in_width = in_size
        self.conv_bn = nn.BatchNorm2d(in_channels)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_height * in_width * in_channels, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv_bn(x)
        # x = F.relu(x)
        # x = self.avg_pool(x)
        # classification = x.reshape(1, self.num_classes)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        classification = self.fc3(x)

        return classification
