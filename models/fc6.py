import torch
import torch.nn as nn
import numpy as np
import random
from opts import randomseed

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)


# sixth fully connected layer

class my_fc6(nn.Module):
    def __init__(self):
        super(my_fc6, self).__init__()
        self.fc = nn.Linear(8192,4096)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        return x