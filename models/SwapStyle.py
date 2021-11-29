import torch
import torch.nn as nn


class SwapStyle(nn.Module):
    def __init__(self, eps=1e-6, detach_flag=0):
        super().__init__()
        self.detach_flag = detach_flag
        self.eps = eps

    def forward(self, x):
        # NxCxHxW
        batch_size = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()

        if self.detach_flag:
            mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        perm = torch.randperm(batch_size)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu2
        sig_mix = sig2

        return x_normed*sig_mix + mu_mix
