from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math

from config import Config

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout_rate: float, max_lenght: int = 1000) -> None:
        """
        `embed_dim`: size of hiiden representation.
        `max_legnth`: max size of input sequence.
        """
        super().__init__()

        den = torch.exp(- torch.arange(0, embed_dim, 2) * math.log(10000) / embed_dim)
        pos = torch.arange(0, max_lenght).reshape(max_lenght, 1)
        self.pos_embedding = torch.zeros((max_lenght, embed_dim))
        self.pos_embedding[:, 0::2] = torch.sin(pos * den)
        self.pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = self.pos_embedding.unsqueeze(-2)
        self.pos_embedding = nn.Parameter(self.pos_embedding)

        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.tensor:
        """
        `x`: (L, B, d)

        `output`: (L, B, d)
        """
        return self.dropout(x + self.pos_embedding[:x.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        """
        `tokens`: (L, B)

        `output`: (L, B, emb_size)
        """
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout_rate=dropout
        )

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor
    ):
        """
        `src`: (L, B, emb_size)
        `trg`: (T, B, emb_size)
        `src_mask`: (L, L)
        `tgt_mask`: (T, T)
        `src_padding_mask`: (L, B)
        `tgt_padding_mask`: (L, B)
        `memory_key_padding_mask`: ???

        `output`: (T, B, tgt_vocab_size)
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        """
        `src`: (L, B)
        `src_mask`: (L, L)

        `output`: (L, B, emb_size)
        """
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)),
            src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        """
        `tgt`: (T, B)
        `memory`: (L, B, emb_size) - output of encoder layers
        `tgt_mask`: (T, T)

        `output`: (T, B, tgt_vocab_size)
        """
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask
        )


def load_model(
    config: Config, source_vocab_size: int, target_vocab_size: int, model_weights: str | None = None
) -> Seq2SeqTransformer:
    transformer = Seq2SeqTransformer(
        config.encoder_layers_num,
        config.decoder_layer_num,
        config.embedding_size,
        config.number_of_heads,
        source_vocab_size,
        target_vocab_size,
        config.feed_forward_hidden_size,
        config.dropout_rate
    )
    if model_weights is None:
        generator = torch.Generator()
        generator.manual_seed(228)
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, generator=generator)
    else:
        transformer.load_state_dict(torch.load(model_weights))

    return transformer
