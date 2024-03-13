import torch
from torch import Tensor

from model import Seq2SeqTransformer
from train import generate_square_subsequent_mask
from tokenizer import BOS_IDX, EOS_IDX

@torch.no_grad()
def beam_search_smater(
    model: Seq2SeqTransformer,
    width: int,
    length_penalty: float,
    temp: float,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    start_symbol: int,
    device: torch.device
) -> Tensor:
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory: Tensor = model.encode(src, src_mask).to(device)
    ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device) # (lenght, hypotesis_count = width)
    top_log_probs = torch.zeros(width, 1, device=device)
    finished: list[tuple[float, Tensor]] = []
    for step in range(max_len - 1):
        # print("STEP", step)
        # print(ys)
        tgt_mask = generate_square_subsequent_mask(ys.size(0), device=device).type(torch.bool)
        # print(tgt_mask.shape)
        # print(ys.shape)
        # print(memory.shape)
        out = model.decode(ys, memory.repeat(1, ys.size(1), 1), tgt_mask) # (lenght, hypotesis_count, embed_dim)
        out = out.transpose(0, 1) # (hypotesis_count, lenght, embed_dim)
        log_prob = torch.nn.functional.log_softmax(model.generator(out[:, -1, :]) / temp, dim=1) * (step+1)**length_penalty # (hypotesis_count, vocab_size)
        best_log_prob, best_tokens = log_prob.topk(width, dim=1, largest=True, sorted=True)

        # print(best_log_prob, best_tokens)
        # top_log_probs: (hypotesis_count, 1)
        # best_log_prob: (hypotesis_count, width)
        # top_log_probs + best_log_prob: (hypotesis_count, width)
        # (top_log_probs + best_log_prob)[A, B] - if to batch A add token B
        next_log_prob = top_log_probs + best_log_prob
        # print(next_log_prob)
        # torch.argsort cannot to dim=None, so ravel()
        next_top_idx = next_log_prob.ravel().argsort()
        if step > 0:
            next_batch_idx = next_top_idx // width
            next_token_idx = next_top_idx % width
        else:
            next_batch_idx = torch.zeros(width, dtype=torch.long)
            next_token_idx = torch.arange(width, dtype=torch.long)
        # print(next_token_idx)
        # print(next_batch_idx)
        next_batches: list[Tensor] = []
        for batch_idx, token_idx in zip(next_batch_idx, next_token_idx):
            # print('IN FOR', batch_idx, token_idx)
            next_token = best_tokens[batch_idx, token_idx]
            # print(next_token)
            next_batch = torch.cat((ys[:, batch_idx], next_token.unsqueeze(0))).unsqueeze(1) # (lenght, 1)
            if next_token.item() == EOS_IDX:
                finished.append(
                    (
                        next_log_prob[batch_idx, token_idx],
                        next_batch
                    )
                )
            else:
                top_log_probs[len(next_batches)] = next_log_prob[batch_idx, token_idx].item()
                next_batches.append(next_batch)
                if len(next_batches) >= width:
                    break
        if len(next_batches) == 0:
            break
        ys = torch.cat(next_batches, dim=1)
        if len(finished) > width:
            break
    for i in range(width):
        finished.append(
            (
                top_log_probs[i, 0].item(),
                ys[:, i]
            )
        )
    return max(finished, key=lambda x: x[0])[1]


@torch.no_grad()
def beam_search_v_ebalo(
    model: Seq2SeqTransformer,
    width: int,
    length_penalty: float,
    temp: float,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    start_symbol: int,
    device: torch.device
) -> Tensor:
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory: Tensor = model.encode(src, src_mask).to(device)
    hypotesis: list[tuple[float, Tensor]] = [
       (0, torch.full((1, 1), start_symbol, dtype=torch.long, device=device))
    ]
    finished: list[tuple[float, Tensor]] = []
    for step in range(max_len - 1):
        new_hypotesis: list[tuple[float, Tensor]] = []
        for score, hyp in hypotesis:
            tgt_mask = generate_square_subsequent_mask(hyp.size(0), device=device).type(torch.bool)
            out = model.decode(hyp, memory, tgt_mask)
            out = out.transpose(0, 1)
            log_prob = torch.nn.functional.log_softmax(model.generator(out[:, -1, :]).squeeze(0) / temp, dim=0) / (step+1)**length_penalty
            best_log_prob, best_tokens = log_prob.topk(width, dim=0, largest=True, sorted=True)
            for log, token in zip(best_log_prob, best_tokens):
                pair = (
                    score + log.item(),
                    torch.cat((hyp, token.unsqueeze(0).unsqueeze(1)))
                )
                if token.item() == EOS_IDX:
                    finished.append(pair)
                else:
                    new_hypotesis.append(pair)
        if len(finished) >= 2 * width:
            break
        new_hypotesis.sort(key=lambda x: x[0])
        hypotesis = new_hypotesis[:width]
    if len(finished) < 2 * width:
        finished += hypotesis
    return finished