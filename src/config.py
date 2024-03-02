from dataclasses import dataclass, field
import typing as tp

@dataclass
class ModelConfig:
    source_vocab_size: int =16000,
    target_vocab_size: int = 16000,
    max_len: int = 256,
    N: int = 6,
    embed_dim: int = 256,
    fc_dim: int = 1024,
    heads: int = 4,
    dropout_rate: float = 0.15


@dataclass
class MegaConfig:
    model_name: str
    batch_size: int
    max_len: int
    model_params: ModelConfig = field(default_factory=lambda: ModelConfig())
    
