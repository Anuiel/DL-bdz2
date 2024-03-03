import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(size), requires_grad=True)

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms

        return self.gamma.unsqueeze(0).unsqueeze(1) * x_norm