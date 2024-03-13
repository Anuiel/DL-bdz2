import subprocess
import tempfile

import torch
from torch import Tensor

from model import Seq2SeqTransformer
from tokenizer import TextTransform, EOS_IDX, BOS_IDX
from train import generate_square_subsequent_mask
from config import Config
from dataset import Multi228k


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    start_symbol: int,
    device: torch.device
) -> Tensor:
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = generate_square_subsequent_mask(ys.size(0), device=device).type(torch.bool)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat(
            [
                ys,
                torch.ones(1, 1).type_as(src.data).fill_(next_word)
            ], 
            dim=0
        )
        if next_word == EOS_IDX:
            break
    return ys


def translate(
    model: Seq2SeqTransformer,
    src_sentence: str,
    text_transform: TextTransform,
    config: Config
):
    model.eval()
    src = text_transform.text_transform[config.source_language](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
    tgt_tokens: Tensor = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 20, start_symbol=BOS_IDX, device=config.device
    ).flatten()
    return " ".join(
        text_transform.vocab_transform[config.target_language].lookup_tokens(list(tgt_tokens.cpu().numpy()))
    ).replace("<bos>", "").replace("<eos>", "").replace("<unk>", "")


def get_bleu_score(
    model: Seq2SeqTransformer,
    val_dataset: Multi228k,
    text_transform: TextTransform,
    config: Config,
):
    model.eval()
    with tempfile.NamedTemporaryFile('w') as tmp_file:
        for source, _ in val_dataset:
            target = translate(model, source, text_transform, config)
            tmp_file.write(target + '\n')
        tmp_file.seek(0)
        subprocess_dict = subprocess.run(
            f'cat {tmp_file.name} | sacrebleu data/val.de-en.en --tokenize none --width 2 -b',
            shell=True,
            capture_output=True
        )
    return float(subprocess_dict.stdout.decode())


def eval_test(
    model: Seq2SeqTransformer,
    test_dataset: Multi228k,
    output_file_name: str,
    text_transform: TextTransform,
    config: Config
):
    with open(str(config.path_to_data / output_file_name), 'w') as output_file:
        for source, _ in test_dataset:
            target = translate(model, source, text_transform, config)
            output_file.write(target + '\n')
