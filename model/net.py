import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv4 = nn.Conv2d(16, 64, kernel_size=3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)

        self.dropout1 = nn.Dropout(0.1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.bn1(self.conv2(x)),2))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.bn2(self.conv4(x)),2))
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.fc3(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)