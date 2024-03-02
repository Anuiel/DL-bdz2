from enum import Enum, auto
from typing import Callable, Generator

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from src.transformer.transformer_net import Transformer

class TrainMode(Enum):
    TRAIN = auto()
    EVAL = auto()


def run_epoch(
    model: Transformer,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    scheduler: LRScheduler | None,
    mode: TrainMode,
    device: torch.device
) -> Generator[float, None, None]:
    if mode == TrainMode.TRAIN:
        model.train()
    elif mode == TrainMode.EVAL:
        model.eval()

    mask = Transformer.causal_mask(256).unsqueeze(0).to(device)
    iterator = tqdm(enumerate(dataloader))
    for i, batch in iterator:

        labels = batch.target_tokens.to(device)

        out = model.forward(
            batch.encoder_input.to(device),
            batch.decoder_input.to(device),
            ~batch.encoder_mask.unsqueeze(1).unsqueeze(1).to(device),
            ~(batch.decoder_mask.unsqueeze(1).to(device) & mask).unsqueeze(1)
        )

        loss = loss_func(out, labels)

        if mode == TrainMode.TRAIN:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        yield loss.detach().cpu().item()
        iterator.set_postfix({'loss': loss.detach().cpu().item()})
        del loss
