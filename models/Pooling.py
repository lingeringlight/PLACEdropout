import torch
from torch import nn as nn
import torch.nn.functional as F

class GeMPooling(nn.Module):
    def __init__(self, p=1, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class AddAvgMaxPooling(nn.Module):
    def __init__(self, kernel_size=1, stride=1):
        super(AddAvgMaxPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        max_output = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        avg_output = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        output = 0.5 * (max_output + avg_output)
        return output


class CatAvgMaxPooling(nn.Module):
    def __init__(self, kernel_size=1, stride=1):
        super(CatAvgMaxPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        max_output = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        avg_output = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        output = torch.cat([max_output, avg_output], 1)
        return output
