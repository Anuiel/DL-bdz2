from pathlib import Path

import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from sacrebleu.metrics import BLEUScore, BLEU

from src.transformer import Transformer


def top_sampling(logits: torch.Tensor, k: int = 50, p: float = 0.92):
    if k > 0:
        indices_to_remove = logits < torch.topk(logits, k)[0][-1, None]
        logits[indices_to_remove] = -torch.inf
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = 0
    
    sorted_logits[sorted_indices_to_remove] = -torch.inf
    logits = torch.gather(sorted_logits, 0, sorted_indices.argsort(-1))
    pred_token = torch.multinomial(torch.nn.functional.softmax(logits, -1), 1)
    return pred_token


def greedy_sampling(logits: torch.Tensor):
    return logits.argmax(dim=-1)

def inference(
    model: Transformer,
    source_tokens: torch.Tensor,
    source_mask: torch.Tensor,
    target_tokenizer: SentencePieceProcessor,
    max_len: int,
    device: torch.device
) -> str:
    model.eval()

    bos_id = target_tokenizer.bos_id()
    eos_id = target_tokenizer.eos_id()

    encoder_output = model.encode(source_tokens, source_mask)

    decoder_input = torch.empty(1, 1).fill_(bos_id).type_as(source_tokens).to(device)

    while True:
        if decoder_input.size(1) >= max_len:
            break

        decoder_mask = ~model.causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(decoder_input, decoder_mask, encoder_output, source_mask)
        logits = model.generate(out[:, -1].squeeze(0))
        next_token = top_sampling(logits, k=20, p=0.92)

        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source_tokens).fill_(next_token.item()).to(device)], dim=1
        )
        if next_token == eos_id:
            break

    return target_tokenizer.Decode(list(map(int, decoder_input.squeeze(0).detach().cpu().numpy())))


def blue_score(
    loader: DataLoader,
    model: Transformer,
    target_tokenizer: SentencePieceProcessor,
    device: torch.device
) -> BLEUScore:
    scorer = BLEU(
        tokenize=None
    )

    ground_truth = []
    predicted = []
    for batch in tqdm(loader):
        source_tokens = batch.encoder_input.to(device).unsqueeze(0)
        encoder_mask = ~batch.encoder_mask.unsqueeze(0).unsqueeze(1).to(device)
        target = inference(model, source_tokens, encoder_mask, target_tokenizer, 128, device)
        ground_truth.append(batch.original_text['target_text'][0])
        predicted.append(target)
    return scorer.corpus_score(predicted, [ground_truth])

def eval_test(
    path_to_file: Path,
    loader: DataLoader,
    model: Transformer,
    target_tokenizer: SentencePieceProcessor,
    device: torch.device,
):
    with open(path_to_file, 'w') as output_file: 
        for batch in tqdm(loader):
            source_tokens = batch.encoder_input.to(device).unsqueeze(0)
            encoder_mask = ~batch.encoder_mask.unsqueeze(0).unsqueeze(1).unsqueeze(1).to(device)

            target = inference(model, source_tokens, encoder_mask, target_tokenizer, 128, device)
            output_file.write(target + '\n')
