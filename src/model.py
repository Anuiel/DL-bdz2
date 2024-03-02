import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()

