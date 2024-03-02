import math

import torch
from torch import nn

# TODO: Rewrite this whole thing without loops
# like https://www.youtube.com/watch?v=ISNdQcPhsts
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float) -> None:
        """
        `embed_dim`: dimention of input and output vectors
        `num_heads`: amount of attention heads
        """
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        `query`: (B, L, d)
        `key`: (B, L, d)
        `value`: (B, L, d)
        `mask`: None | (1, L) | (L, L)

        `output`: (B, L, d), (B, L, L)
        """
        B = query.shape[0]

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        value = value.reshape(B, -1, self.num_heads, self.head_dim)
        query = query.reshape(B, -1, self.num_heads, self.head_dim)
        key = key.reshape(B, -1, self.num_heads, self.head_dim)

        # (Q^T @ K) but batched
        # (B, L_q, heads, head_dim) * (B, L_k, heads, head_dim) --> (B, heads, L_q, L_k) 
        # My own awesome research shows that einsum almost as fast as regular method with transpose + @
        logits = torch.einsum("bqhd,bkhd->bhqk", [query, key])
        if mask is not None:
            logits = logits.masked_fill(mask, -torch.inf)

        attention = torch.softmax(logits / (self.d_model**(0.5)), dim=3)

        # (B, heads, L_q, L_k) * (B, L_v, heads, head_dim) -> (B, L_q, heads, head_dim)
        # L_k = L_v
        out = torch.einsum("bhqv,bvhd->bqhd", [attention, value]).reshape(B, -1, self.d_model)

        return out
