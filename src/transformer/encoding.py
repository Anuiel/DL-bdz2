import math

import torch
from torch import nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, max_lenght: int, embed_dim: int, dropout_rate: float) -> None:
        """
        `max_legnth`: max size of input sequence.
        `embed_dim`: size of hiiden representation.
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout_rate)
        self.pos_encoding = torch.zeros(max_lenght, embed_dim)

        sequence_index = torch.arange(0, max_lenght, dtype=torch.float).unsqueeze(1)
        freq = torch.exp(
            -math.log(10000) * torch.arange(0, embed_dim, 2, dtype=torch.float) / embed_dim
        ).unsqueeze(0)

        arguments = sequence_index * freq

        self.pos_encoding[:, 0::2] = torch.sin(arguments)
        self.pos_encoding[:, 1::2] = torch.cos(arguments)
        self.pos_encoding = self.pos_encoding.unsqueeze(0)
        self.pos_encoding = nn.Parameter(self.pos_encoding, requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.tensor:
        """
        `x`: (B, L, d) -> (B, L, d)
        """
        return self.dropout(x + self.pos_encoding[:, :x.shape[1]])