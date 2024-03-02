import torch
from torch.distributions.categorical import Categorical
from sentencepiece import SentencePieceProcessor

from src.transformer import Transformer

def top_k_sampling(logits: torch.Tensor, k: int = 50):
    reduced_logist, args = logits.topk(sorted=False, k=k)
    return args[Categorical(logits=reduced_logist).sample()]

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
        next_token = top_k_sampling(logits)

        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source_tokens).fill_(next_token.item()).to(device)], dim=1
        )
        if next_token == eos_id:
            break

    return target_tokenizer.Decode(list(map(int, decoder_input.squeeze(0).detach().cpu().numpy())))