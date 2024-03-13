import torch
from torch import Tensor

from model import Seq2SeqTransformer
from train import generate_square_subsequent_mask
from tokenizer import BOS_IDX, EOS_IDX

@torch.no_grad()
def beam_search(
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