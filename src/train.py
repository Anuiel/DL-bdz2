from typing import Callable

import torch
from torch.utils.data import DataLoader
import wandb

from model import Seq2SeqTransformer
from tokenizer import PAD_IDX


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device: torch.device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(
    model: Seq2SeqTransformer,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    cumulative_index: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
) -> float:
    model.train()
    losses = 0
    for src, tgt in data_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device=device)

        logits = model.forward(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_function(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss = loss.item()
        if (cumulative_index + 1) % 25 == 0:
            wandb.log({
                "loss": loss,
                "step": cumulative_index * data_loader.batch_size
            }
        )
        cumulative_index += 1
        losses += loss

    return losses / len(list(data_loader)), cumulative_index

@torch.no_grad()
def evaluate(
    model: Seq2SeqTransformer,
    data_loader: DataLoader,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> float:
    model.eval()
    losses = 0
    for src, tgt in data_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device=device)

        logits = model.forward(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_function(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(data_loader))