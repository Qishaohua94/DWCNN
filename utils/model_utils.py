import torch
import torch.nn as nn
import torch.nn.functional as F


class Max_Viewpooling(nn.Module):
    def __init__(self):
        super(Max_Viewpooling, self).__init__()

    def forward(self, x):
        x, _ = torch.max(x, 1)
        x = x.view(x.size(0), -1)
        return x

