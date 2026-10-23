"""Post-norm encoder/decoder assembled with PyTorch multi-head attention blocks."""

import torch
from torch import nn

from .data import PAD


def causal_mask(length, device=None):
    if length < 1:
        raise ValueError("Sequence length must be positive")
    return torch.triu(
        torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1
    )


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, width, max_length):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width, padding_idx=PAD)
        self.position = nn.Embedding(max_length, width)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.position.num_embeddings:
            raise ValueError("Expected batch x sequence within the positional limit")
        return (
            self.token(tokens)
            + self.position(torch.arange(tokens.shape[1], device=tokens.device))[
                None, :, :
            ]
        )


class EncoderBlock(nn.Module):
    def __init__(self, width, heads, ff_width, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ff_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_width, width),
        )

    def forward(self, source, padding):
        attended, _ = self.attention(
            source, source, source, key_padding_mask=padding, need_weights=False
        )
        hidden = self.norm1(source + attended)
        return self.norm2(hidden + self.ffn(hidden))


class DecoderBlock(nn.Module):
    def __init__(self, width, heads, ff_width, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.norm1, self.norm2, self.norm3 = (
            nn.LayerNorm(width),
            nn.LayerNorm(width),
            nn.LayerNorm(width),
        )
        self.ffn = nn.Sequential(
            nn.Linear(width, ff_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_width, width),
        )

    def forward(self, target, memory, source_padding, target_padding, capture=False):
        attended, _ = self.self_attention(
            target,
            target,
            target,
            attn_mask=causal_mask(target.shape[1], target.device),
            key_padding_mask=target_padding,
            need_weights=False,
        )
        hidden = self.norm1(target + attended)
        attended, weights = self.cross_attention(
            hidden,
            memory,
            memory,
            key_padding_mask=source_padding,
            need_weights=capture,
            average_attn_weights=True,
        )
        hidden = self.norm2(hidden + attended)
        return self.norm3(hidden + self.ffn(hidden)), weights


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        width=64,
        heads=4,
        layers=2,
        ff_width=128,
        dropout=0.1,
        max_length=32,
    ):
        super().__init__()
        if (
            min(vocab_size, width, heads, layers, ff_width, max_length) < 1
            or width % heads
            or not 0 <= dropout < 1
        ):
            raise ValueError(
                "Positive dimensions, divisible head width and dropout in [0,1) required"
            )
        self.source_embedding = TokenAndPositionEmbedding(vocab_size, width, max_length)
        self.target_embedding = TokenAndPositionEmbedding(vocab_size, width, max_length)
        self.encoder = nn.ModuleList(
            [EncoderBlock(width, heads, ff_width, dropout) for _ in range(layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(width, heads, ff_width, dropout) for _ in range(layers)]
        )
        self.output = nn.Linear(width, vocab_size)
        self.max_length = max_length

    def encode(self, source):
        padding = source.eq(PAD)
        if source.ndim != 2 or source.shape[1] < 1 or padding.all(1).any():
            raise ValueError(
                "Each source sequence needs at least one non-padding token"
            )
        memory = self.source_embedding(source)
        for layer in self.encoder:
            memory = layer(memory, padding)
        return memory, padding

    def decode(self, target, memory, source_padding, capture=False):
        if target.ndim != 2 or target.shape[1] < 1 or target.eq(PAD).all(1).any():
            raise ValueError("Each target sequence needs a non-padding prefix")
        hidden = self.target_embedding(target)
        weights = None
        for index, layer in enumerate(self.decoder):
            hidden, weights = layer(
                hidden,
                memory,
                source_padding,
                target.eq(PAD),
                capture and index == len(self.decoder) - 1,
            )
        return self.output(hidden), weights

    def forward(self, source, target):
        memory, padding = self.encode(source)
        return self.decode(target, memory, padding)[0]
