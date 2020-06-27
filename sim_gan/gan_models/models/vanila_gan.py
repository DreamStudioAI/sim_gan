import torch.nn as nn
import torch.nn.functional as F


class VGenerator(nn.Module):
    def __init__(self, ngpu):
        super(VGenerator, self).__init__()
        self.ngpu = ngpu
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 216)

    def forward(self, x):
        x = x.view(-1, 100)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VDiscriminator(nn.Module):
    def __init__(self, ngpu):
        super(VDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.fc1 = nn.Linear(216, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x