import torch
from torch import nn

from src.transformer.attention import MultiHeadAttention
from src.transformer.encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, fc_dim: int, num_heads: int, dropout_rate: float):
        """
        `embed_dim`: dimention of input and output vectors
        `fc_dim`: dimention of hidden state in FeedForward step
        `num_heads`: heads in `MultiHeadAttention`
        `dropout_rate`: dropout_rate
        """
        super().__init__()

        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, fc_dim),
            nn.GELU(),
            nn.Linear(fc_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        `inputs`: (B, L, d)
        `mask`: (L, L)

        `output.0`: (B, L, d)
        `output.1`: (B, L, L)
        """
        attention = self.self_attention(
            query=inputs,
            key=inputs, 
            value=inputs,
            mask=mask
        )

        outputs = inputs + self.norm1(attention)
        outputs = outputs + self.norm2(self.feedforward(outputs))
        return outputs


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, fc_dim: int, num_heads: int, dropout_rate: float):
        """
        `embed_dim`: dimention of input and output vectors
        `fc_dim`: dimention of hidden state in FeedForward step
        `num_heads`: heads in `MultiHeadAttention`
        `dropout_rate`: dropout_rate
        """
        super().__init__()

        self.masked_self_attention = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, fc_dim),
            nn.GELU(),
            nn.Linear(fc_dim, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: torch.Tensor,
        encoder_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        `inputs`: (B, L, d)
        `encoder_output`: (B, L, d)
        `mask`: None | (L, L)
        `encoder_mask`: None | (L, L)

        `output.0`: (B, L, d)
        `output.1`: (B, L, L)
        """
        attention = self.masked_self_attention(
            query=inputs,
            key=inputs, 
            value=inputs,
            mask=mask
        )
        outputs = inputs + self.norm1(attention)
        attention = self.self_attention(
            query=outputs,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        outputs = outputs + self.norm2(attention)
        outputs = outputs + self.norm3(self.feedforward(outputs))
        return outputs


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, embed_dim: int, vocab_size: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        return self.proj(x)
