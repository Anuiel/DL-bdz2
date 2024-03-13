from dataclasses import dataclass, field
from pathlib import Path

import torch

@dataclass
class Config:
    path_to_data: Path = Path('data')
    source_language: str = 'de'
    target_language: str = 'en'

    embedding_size: int = 512
    number_of_heads: int = 8
    feed_forward_hidden_size: int = 512
    encoder_layers_num: int = 5
    decoder_layer_num: int = 5
    dropout_rate: float = 0.1

    n_epoch: int = 6
    batch_size: int = 16

    learning_rate: float = 5e-4

    label_smooting: float = 0.0

    device: torch.device = torch.device('cuda')
