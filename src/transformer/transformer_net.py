import torch
from torch import nn

from src.transformer.transformer_layers import EncoderLayer, DecoderLayer, Generator
from src.transformer.encoding import Embeddings, PositionalEncoding

class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        max_len: int = 256,
        N: int = 6,
        embed_dim: int = 512,
        fc_dim: int = 2028,
        heads: int = 8,
        dropout_rate: float = 0.1
    ) -> None:
        super().__init__()

        self.source_embeddings = nn.Sequential(
            Embeddings(source_vocab_size, embed_dim),
            PositionalEncoding(max_len, embed_dim, dropout_rate)
        )
        self.target_embeddings = nn.Sequential(
            Embeddings(target_vocab_size, embed_dim),
            PositionalEncoding(max_len, embed_dim, dropout_rate)
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim, fc_dim, heads, dropout_rate)
                for _ in range(N)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(embed_dim, fc_dim, heads, dropout_rate)
                for _ in range(N)
            ]
        )

        self.generator = Generator(embed_dim, target_vocab_size)

    @staticmethod
    def causal_mask(size: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.bool)
        return mask == 0

    def encode(self, encoder_tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        `input_sequence`: (B, L)
        `mask`: (B, L) - padding mask

        `return`: (B, L, d)
        """
        embeddings = self.source_embeddings(encoder_tokens)
        for layer in self.encoder_layers:
            embeddings = layer(embeddings, mask)
        return embeddings
    
    def decode(
        self,
        decoder_tokens: torch.Tensor,
        decoder_mask: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor
    ) -> torch.Tensor:
        decoder_embeddings = self.target_embeddings(decoder_tokens)
        for layer in self.decoder_layers:
            decoder_embeddings = layer(decoder_embeddings, encoder_output, decoder_mask, encoder_mask)
        return decoder_embeddings
    
    def generate(self, decoder_embeddings: torch.Tensor) -> torch.Tensor:
        return self.generator(decoder_embeddings)

    def forward(
        self,
        encoder_tokens: torch.Tensor,
        decoder_tokens: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor
    ) -> torch.Tensor:
        encoder_output = self.encode(encoder_tokens, encoder_mask)
        predicted = self.decode(decoder_tokens, decoder_mask, encoder_output, encoder_mask)
        return self.generate(predicted)
