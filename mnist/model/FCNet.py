import numpy as np
import torch
from torch import nn

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, data):
        data = self.features(data)
        return data


