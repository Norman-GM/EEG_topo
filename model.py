import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DumbNetwork(nn.Module):

    def __init__(self):
        super(DumbNetwork, self).__init__()
        self.fc1 = nn.Linear(310, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
