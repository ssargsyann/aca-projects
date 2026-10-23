"""Greedy autoregressive decoding, with correctly labeled character/EOS attention rows."""

import datetime as dt
import re

import torch

from .data import BOS, EOS, PAD, SOURCE_LENGTH, VOCAB, encode


def is_iso(text):
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text) is None:
        return False
    try:
        return dt.date.fromisoformat(text).isoformat() == text
    except ValueError:
        return False


@torch.no_grad()
def generate(model, sources, device="cpu", max_tokens=12, attention=False):
    if not sources:
        return []
    if not 1 <= max_tokens <= model.max_length:
        raise ValueError("Generation budget exceeds positional capacity or is empty")
    if attention and len(sources) != 1:
        raise ValueError("Attention export expects exactly one source")
    model.eval()
    source = torch.stack([encode(text, SOURCE_LENGTH) for text in sources]).to(device)
    memory, padding = model.encode(source)
    prefixes = torch.full((len(sources), 1), BOS, device=device, dtype=torch.long)
    completed = torch.zeros(len(sources), device=device, dtype=torch.bool)
    tokens = [[] for _ in sources]
    rows = []
    for _ in range(max_tokens):
        logits, weights = model.decode(prefixes, memory, padding, capture=attention)
        next_tokens = logits[:, -1].argmax(-1)
        if attention:
            rows.append(weights[0, -1, : len(sources[0])].cpu().tolist())
        for index, value in enumerate(next_tokens.cpu().tolist()):
            if not completed[index]:
                tokens[index].append(value)
        completed |= next_tokens.eq(EOS)
        next_tokens = torch.where(completed, PAD, next_tokens)
        prefixes = torch.cat([prefixes, next_tokens[:, None]], dim=1)
        if completed.all():
            break
    output = []
    for index, ids in enumerate(tokens):
        labels = [VOCAB[value] for value in ids]
        text = "".join(label for label in labels if label != "<EOS>")
        item = {
            "source": sources[index],
            "prediction": text,
            "terminated": bool(ids and ids[-1] == EOS),
            "valid_iso": is_iso(text),
            "token_ids": ids,
        }
        if attention:
            item.update(
                attention=rows,
                attention_rows=labels,
                attention_columns=list(sources[index]),
                attention_scope="last_decoder_layer_mean_over_heads_during_greedy_steps",
            )
        output.append(item)
    return output
