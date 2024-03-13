import typing as tp
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab

from dataset import Multi228k


PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_SYMBOLS = ['<pad>', '<unk>', '<bos>', '<eos>']


class TextTransform:
    def __init__(self, path_to_data: Path, source_language: str, target_language: str) -> None:
        self.source_language = source_language
        self.target_language = target_language

        self.token_transform: dict[str, tp.Callable[[str], list[str]]] = {}
        self.vocab_transform: dict[str, Vocab] = {}
        self.text_transform: dict[str, tp.Callable[[str], torch.Tensor]] = {}

        self.token_transform[source_language] = get_tokenizer(None)
        self.token_transform[target_language] = get_tokenizer(None)

        def yield_tokens(data_iter: tp.Iterable[tuple[str, str]], language: str):
            language_index = {source_language: 0, target_language: 1}

            for data_sample in data_iter:
                yield self.token_transform[language](data_sample[language_index[language]])

        for ln in [source_language, target_language]:
            train_iter = Multi228k(path_to_data, ['train', 'val', 'test1'], source_language, target_language)
            self.vocab_transform[ln] = build_vocab_from_iterator(
                yield_tokens(train_iter, ln),
                min_freq=3,
                specials=SPECIAL_SYMBOLS,
                special_first=True
            )

        for ln in [source_language, target_language]:
            self.vocab_transform[ln].set_default_index(UNK_IDX)
            self.text_transform[ln] = self.sequential_transforms(
                self.token_transform[ln],
                self.vocab_transform[ln],
                self.tensor_transform
            )

    @staticmethod
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    @staticmethod
    def tensor_transform(token_ids: list[int]):
        return torch.cat(( 
            torch.tensor([BOS_IDX]),
            torch.tensor(token_ids),
            torch.tensor([EOS_IDX])
        ))

    def collate_fn(self, batch: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform[self.source_language](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform[self.target_language](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch
