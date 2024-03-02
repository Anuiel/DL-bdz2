from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, random_split
import sentencepiece as spm

@dataclass
class LanguagePair:
    source_text: str
    target_text: str


class LanguageDataset(Dataset):
    def __init__(self, source_lang_path: Path, target_lang_path: Path | None = None, limit: int = -1) -> None:
        super().__init__()

        self.only_source = (target_lang_path is None)
        with open(source_lang_path) as source_lang_file:
            self.source_lang = source_lang_file.readlines(limit)

        if not self.only_source:
            with open(target_lang_path) as target_lang_file:
                self.target_lang = target_lang_file.readlines(limit)

            assert len(self.source_lang) == len(self.target_lang)

    def __len__(self) -> int:
        return len(self.source_lang)

    def __getitem__(self, index: int) -> LanguagePair:
        if not self.only_source:
            return LanguagePair(
                source_text=self.source_lang[index].rstrip('\n'),
                target_text=self.target_lang[index].rstrip('\n')
            )
        return LanguagePair(
            source_text=self.source_lang[index].rstrip('\n'),
            target_text=''
        )


@dataclass
class TokenizedLanguagePair:
    # (seq_len,)
    encoder_input: torch.Tensor
    decoder_input: torch.Tensor
    target_tokens: torch.Tensor # expected output of decoder

    # (seq_len,)
    encoder_mask: torch.Tensor # to mask padding
    # (seq_len,)
    decoder_mask: torch.Tensor

    original_text: LanguagePair

    def __str__(self) -> str:
        return "\n".join([
            f"encoder_input: {self.encoder_input.shape}",
            f"decoder_input: {self.decoder_input.shape}",
            f"target_tokens: {self.target_tokens.shape}",
            f"encoder_mask:  {self.encoder_mask.shape}",
            f"decoder_mask:  {self.decoder_mask.shape}"
        ])


class LanguageDatasetTokenized(Dataset):
    def __init__(
        self,
        dataset: LanguageDataset,
        source_tokenizer: spm.SentencePieceProcessor,
        target_tokenizer: spm.SentencePieceProcessor,
        seq_len: int
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.seq_len = seq_len

        self.pad_id: int = self.source_tokenizer.pad_id()
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        language_pair = self.dataset[index]

        source_tokens = self.source_tokenizer.Encode(language_pair.source_text, add_bos=True, add_eos=True)
        target_tokens = self.target_tokenizer.Encode(language_pair.target_text, add_bos=False, add_eos=False)

        source_padding_count = self.seq_len - len(source_tokens)
        target_padding_count = self.seq_len - len(target_tokens) - 1
        if source_padding_count < 0 or target_padding_count < 0:
            raise ValueError('Too large sequence! Maybe increase seq_len?')

        encoder_input = torch.cat((
            torch.tensor(source_tokens, dtype=torch.int64),
            torch.tensor([self.pad_id] * source_padding_count, dtype=torch.int64)
        ))

        decoder_input = torch.cat((
            torch.tensor([self.target_tokenizer.bos_id()]),
            torch.tensor(target_tokens, dtype=torch.int64),
            torch.tensor([self.pad_id] * target_padding_count, dtype=torch.int64)
        ))

        labels = torch.cat((
            torch.tensor(target_tokens, dtype=torch.int64),
            torch.tensor([self.target_tokenizer.eos_id()]),
            torch.tensor([self.pad_id] * target_padding_count, dtype=torch.int64)
        ))
        return TokenizedLanguagePair(
            encoder_input,
            decoder_input,
            labels,
            (encoder_input != self.pad_id).bool(),
            (decoder_input != self.pad_id).bool(),
            language_pair
        )


def get_dataset(
    source_lang_path: Path,
    target_lang_path: Path,
    source_tokenizer_path: Path,
    target_tokenizer_path: Path,
    seq_len: int = 128
) -> LanguageDatasetTokenized:
    source_tokenizer = spm.SentencePieceProcessor()
    source_tokenizer.Load(source_tokenizer_path)

    target_tokenizer = spm.SentencePieceProcessor()
    target_tokenizer.Load(target_tokenizer_path)

    raw_dataset = LanguageDataset(source_lang_path, target_lang_path)

    return LanguageDatasetTokenized(raw_dataset, source_tokenizer, target_tokenizer, seq_len)


def split_train_val(dataset: Dataset, train_size: float = 0.9, random_seed: int = 228) -> tuple[Dataset, Dataset]:
    gen = torch.Generator().manual_seed(random_seed)

    return tuple(random_split(dataset, [train_size, 1 - train_size], generator=gen))