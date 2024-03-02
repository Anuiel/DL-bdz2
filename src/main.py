from pathlib import Path
from dataclasses import asdict
import typing as tp

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, default_collate
from sentencepiece import SentencePieceProcessor
import wandb
from tqdm import tqdm

from src.dataset import LanguageDataset, LanguageDatasetTokenized, split_train_val, TokenizedLanguagePair
from src.tokenizer import load_tokenizers
from src.transformer import Transformer
from src.train import run_epoch, TrainMode
from src.inference import inference, blue_score

PATH_TO_DATA = Path('/home/anuiel/Remote/Anuiel/dl-bdz2/data')
PATH_TO_TOKENIZERS = PATH_TO_DATA / 'tokenizer'
PATH_TO_TEXT = PATH_TO_DATA / 'texts'
PATH_TO_MODEL = PATH_TO_DATA / 'models'



def create_loaders(
    de_sp: SentencePieceProcessor,
    en_sp: SentencePieceProcessor,
    max_len: int,
    batch_size: int,
    train_size: float = 0.95,
    random_seed: int = 228
) -> tuple[DataLoader, DataLoader]:
    def collate(batch: list[TokenizedLanguagePair]) -> TokenizedLanguagePair:
        return TokenizedLanguagePair(**default_collate([asdict(item) for item in batch]))

    raw_dataset = LanguageDataset(PATH_TO_TEXT / 'train.de', PATH_TO_TEXT / 'train.en')
    dataset = LanguageDatasetTokenized(raw_dataset, de_sp, en_sp, max_len)

    train_dataset, val_dataset = split_train_val(dataset, train_size=train_size, random_seed=random_seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, shuffle=False, collate_fn=collate)
    inference_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate)

    return train_loader, val_loader, inference_loader


def load_model(
    source_vocab_size: int,
    target_vocab_size: int,
    max_len: int,
    N: int,
    embed_dim: int,
    fc_dim: int,
    heads: int,
    dropout_rate: float,
    device: torch.device,
    pre_train: Path | None = None
) -> Transformer:
    
    model = Transformer(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        max_len=max_len,
        N=N,
        embed_dim=embed_dim,
        fc_dim=fc_dim,
        heads=heads,
        dropout_rate=dropout_rate
    ).to(device)

    if pre_train is not None:
        model.load_state_dict(torch.load(pre_train))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform(p)

    def rate(step, model_size, factor, warmup):
        if step == 0:
            step = 1
        return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        )

    optim = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optim, lr_lambda=lambda step: rate(step, embed_dim, 1.0, 3000)
    )

    return model, optim, lr_scheduler


def load_everything(
    train_size: float,
    batch_size: int,
    source_vocab_size: int,
    target_vocab_size: int,
    max_len: int,
    N: int,
    embed_dim: int,
    fc_dim: int,
    heads: int,
    dropout_rate: float,
    label_smoothing: float,
    random_seed: int = 228,
    model_pre_train: Path | None = None
) -> dict[str, tp.Any]:

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    en_sp, de_sp = load_tokenizers(source_vocab_size, PATH_TO_TEXT, PATH_TO_TOKENIZERS)
    train_loader, val_loader, inference_loader = create_loaders(
        de_sp, en_sp, max_len, batch_size, train_size, random_seed=random_seed
    )
    model, optim, scheduler = load_model(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        max_len=max_len,
        N=N,
        embed_dim=embed_dim,
        fc_dim=fc_dim,
        heads=heads,
        dropout_rate=dropout_rate,
        device=device,
        pre_train=model_pre_train
    )
    def loss_func(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            predicted.view(-1, target_vocab_size),
            target.view(-1),
            label_smoothing=label_smoothing,
            ignore_index=0
        )
    return {
        'tokenizers': (de_sp, en_sp),
        'model': (model, optim, scheduler),
        'loader': (train_loader, val_loader, inference_loader),
        'loss_func': loss_func,
        'device': device
    }

def main(mode: TrainMode, commit_hash: str, pretrain_path: Path | None = None):
    stuff = load_everything(
        train_size=0.97,
        batch_size=24,
        source_vocab_size=30000,
        target_vocab_size=30000,
        max_len=512,
        N=5,
        embed_dim=256,
        fc_dim=1024,
        heads=8,
        dropout_rate=0.0,
        label_smoothing=0.1,
        random_seed=228,
        model_pre_train=pretrain_path
    )
    de_sp, en_sp = stuff['tokenizers']
    model, optim, scheduler = stuff['model']
    train_loader, val_loader, inference_loader = stuff['loader']
    loss_func = stuff['loss_func']
    device = stuff['device']

    if mode == TrainMode.TRAIN:

        wandb.login()
        wandb.init(
            project="bdz2",
            config={
                'commit-hash': commit_hash
            }
        )

        train_iteration = 0
        val_iteration = 0
        for _ in range(8):
            torch.cuda.empty_cache()
            for i, x in enumerate(run_epoch(model, train_loader, optim, loss_func, scheduler, TrainMode.TRAIN, device)):
                train_iteration += 1
                wandb.log({
                    'train loss': x,
                    'train step': train_iteration
                })

            for j, y in enumerate(run_epoch(model, val_loader, optim, loss_func, scheduler, TrainMode.EVAL, device)):
                val_iteration += 1
                wandb.log({
                    'val loss': y,
                    'val step': val_iteration
                }) 
        torch.save(model.state_dict(), PATH_TO_MODEL / f'{wandb.run.id}.pth')

    elif mode == TrainMode.EVAL:
        print(blue_score(inference_loader, model, en_sp, device))
        # test_dataset = LanguageDataset(PATH_TO_DATA / 'test1.de-en.de')
        # test_dataset_tokenized = LanguageDatasetTokenized(test_dataset, de_sp, en_sp, 256)

        # with open(PATH_TO_DATA / 'test1.de-en.en', 'w') as output_file: 
        #     for batch in tqdm(test_dataset_tokenized):
        #         source_tokens = batch.encoder_input.to(device).unsqueeze(0)
        #         encoder_mask = ~batch.encoder_mask.unsqueeze(0).unsqueeze(1).unsqueeze(1).to(device)

        #         target = inference(model, source_tokens, encoder_mask, en_sp, 128, device)
        #         output_file.write(target + '\n')
