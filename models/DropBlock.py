import torch
import torch.nn.functional as F
from torch import nn


class DropBlock2D(nn.Module):

    def __init__(self):
        super(DropBlock2D, self).__init__()

    def forward(self, input, keep_prob=0.9, block_size=7):
        if not self.training or keep_prob == 1:
            return input
        gamma = (1. - keep_prob) / block_size ** 2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma)
        Msum = F.conv2d(M,
                        torch.ones((input.shape[1], 1, block_size, block_size)).to(device=input.device,
                                                                                   dtype=input.dtype),
                        padding=block_size // 2,
                        groups=input.shape[1])
        torch.set_printoptions(threshold=5000)
        mask = (Msum < 1).to(device=input.device, dtype=input.dtype)
        return input * mask * mask.numel() /mask.sum()