from timeit import default_timer as timer
from dataclasses import asdict
import click
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from config import Config
from model import load_model
from dataset import Multi228k
from tokenizer import TextTransform, PAD_IDX, BOS_IDX
from train import train_epoch, evaluate
from inference import get_bleu_score, eval_test
from beam_search import beam_search_v_ebalo


RANDOM_SEED = 228
torch.backends.cudnn.deterministic = True
random.seed(RANDOM_SEED + 1)
np.random.seed(RANDOM_SEED + 2)
torch.manual_seed(RANDOM_SEED + 3)
torch.cuda.manual_seed_all(RANDOM_SEED + 4)


@click.command()
@click.option(
    '-g', '--git-hash', required=True
)
def main(git_hash: str):
    config = Config()

    wandb.init(
        project='bdz2',
        config=asdict(config) | {'git hash': git_hash}
    )
    run_id = wandb.run.id

    train_dataset = Multi228k(config.path_to_data, 'train', config.source_language, config.target_language)
    val_dataset = Multi228k(config.path_to_data, 'val', config.source_language, config.target_language)
    test_dataset = Multi228k(config.path_to_data, 'test1', config.source_language, config.target_language)

    text_transform = TextTransform(config.path_to_data, config.source_language, config.target_language)
    source_vocab_size = len(text_transform.vocab_transform[config.source_language])
    target_vocab_size = len(text_transform.vocab_transform[config.target_language])
    decode_function = lambda model, src, src_mask, max_len, device: beam_search_v_ebalo(
        width=2,
        length_penalty=0.5,
        temp=1,
        model=model,
        src=src,
        src_mask=src_mask,
        max_len=max_len,
        start_symbol=BOS_IDX,
        device=device
    )
    model = load_model(config, source_vocab_size, target_vocab_size).to(config.device)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=config.label_smooting)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim,
        [3, 5],
        0.2
    )

    train_gen = torch.Generator().manual_seed(RANDOM_SEED + RANDOM_SEED)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=text_transform.collate_fn,
        generator=train_gen
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=text_transform.collate_fn,
    )
    cumulative_step = 0
    for epoch in range(1, config.n_epoch + 1):
        torch.cuda.empty_cache()
        start_time = timer()
        _, cumulative_step = train_epoch(model, optim, train_dataloader, loss_function, config.device, cumulative_step)
        torch.cuda.empty_cache()
        end_time = timer()
        val_loss = evaluate(model, val_dataloader, loss_function, config.device)
        bleu = get_bleu_score(model, val_dataset, text_transform, config, decode_function)
        lr_scheduler.step()
        eval_test(model, test_dataset, f'{run_id}-{epoch}.en', text_transform, config, decode_function)
        wandb.log({
            'epoch': epoch,
            'val loss': val_loss,
            'bleu': bleu,
            'epoch time': end_time - start_time
        })

    torch.save(model.state_dict(), config.path_to_data / 'models' / f'{run_id}.pth')


if __name__ == '__main__':
    main()