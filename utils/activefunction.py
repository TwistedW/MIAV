import torch
import torch.nn as nn
import torch.nn.functional as F


class EyeLu(nn.Module):
    def __init__(self):
        super(EyeLu, self).__init__()

    def forward(self, x):
        x = x - 0.5
        x = torch.exp(-torch.pow(2, x))
        x = x + 1 - torch.exp(torch.tensor(-1/4))
        return x
