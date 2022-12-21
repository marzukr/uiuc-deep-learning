import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.bn3 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
#         x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
#         x = self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
#         x = self.bn3(x)
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.res_block = ResidualBlock(32,32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.res_block2 = ResidualBlock(64,64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.res_block3 = ResidualBlock(128,128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.res_block4 = ResidualBlock(256,256)
        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(256, 1000)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(1000, NUM_CLASSES)
        
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.batch_norm_3 = nn.BatchNorm2d(128)
        self.batch_norm_4 = nn.BatchNorm2d(256)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch_norm_1(x)
        x = self.res_block(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batch_norm_2(x)
        x = self.res_block2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.batch_norm_3(x)
        x = self.res_block3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.batch_norm_4(x)
        x = self.res_block4(x)
        x = self.avg_pool(x)
        x = x.view(x.size()[0], 256)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


