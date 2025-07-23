import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64], dropout=0.9):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dims=[128, 64], dropout=0.0):
#         super(MLP, self).__init__()
#         layers = []
#         dims = [input_dim] + hidden_dims

#         for i in range(len(hidden_dims)):
#             layers.append(nn.Linear(dims[i], dims[i + 1]))
#             layers.append(nn.BatchNorm1d(dims[i + 1]))
#             layers.append(nn.ReLU())
#             if dropout > 0:
#                 layers.append(nn.Dropout(dropout))

#         layers.append(nn.Linear(hidden_dims[-1], output_dim))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)
