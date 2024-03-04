from pathlib import Path
from typing import Any

from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

# https://github.com/google/sentencepiece/blob/master/doc/options.md


def get_tokenizer_config(
    lang: str,
    vocab_size: int,
    path_to_data: Path,
    path_to_save: Path
) -> dict[str, Any]:
    return {
        'input': path_to_data / f'train.{lang}',
        'model_prefix': path_to_save / f'{lang}-word-{vocab_size}',
        'vocab_size': vocab_size,
        'model_type': 'word',
        'pad_id': 0, 'pad_piece': '[PAD]',
        'unk_id': 1, 'unk_piece': '[UNK]',
        'bos_id': 2, 'bos_piece': '[BOS]',
        'eos_id': 3, 'eos_piece': '[EOS]',
    }


def load_tokenizer(
    lang: str,
    vocab_size: int,
    path_to_data: Path,
    path_to_save: Path   
) -> SentencePieceProcessor:
    cfg = get_tokenizer_config(lang, vocab_size, path_to_data, path_to_save)
    tokenizer_model_name = cfg['model_prefix'].with_suffix('.model')
    if not (tokenizer_model_name).exists():
        SentencePieceTrainer.train(**cfg)
    tokenizer = SentencePieceProcessor()
    tokenizer.load(str(tokenizer_model_name))
    return tokenizer


def load_tokenizers(
    source_vocab_size: int,
    target_vocab_size: int,
    path_to_data: Path,
    path_to_save: Path,
) -> tuple[SentencePieceProcessor, SentencePieceProcessor]:
    return (
        load_tokenizer('de', source_vocab_size, path_to_data, path_to_save),
        load_tokenizer('en', target_vocab_size, path_to_data, path_to_save),
    )