"""
model.py — TinyGPT scaled for Mistral tokenizer (32k vocab).

Architecture differences from the ClaseIV notebook:
- n_embd=256, n_layer=4, n_head=8 (up from 64/2/4)
- vocab_size=32000 (Mistral tokenizer, up from 61 chars)
- Weight tying between token embedding and output head (saves ~8M params)
- GELU activation instead of ReLU (standard in modern GPTs)
- No KV-cache: not needed for training; simplifies the code significantly
"""

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Type
from transformers import AutoTokenizer


MISTRAL_TOKENIZER = "mistralai/Mistral-7B-v0.1"


def load_tokenizer() -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(MISTRAL_TOKENIZER, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


@dataclass
class GPTConfig:
    block_size: int = 128       # max sequence length in tokens
    n_embd: int = 512           # embedding dimension
    n_head: int = 8             # number of attention heads
    n_layer: int = 6            # number of transformer blocks
    dropout: float = 0.2        # 0.1 caused memorisation with <150k tokens
    vocab_size: int = 32000     # Mistral tokenizer vocab size
    bias: bool = True


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention using F.scaled_dot_product_attention.

    Replaces the original per-head Python loop + manual tril mask with a single
    batched call that dispatches to the best available kernel:
      - CUDA  → Flash Attention 2 (when available) or Memory-Efficient Attention
      - MPS   → Metal Performance Shaders fused attention kernel
      - CPU   → math fallback

    All heads are projected together, reshaped, attended in one shot, and
    reshaped back — no Python-level loop, no stored causal mask buffer.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Single projection for Q, K, V across all heads at once
        self.qkv  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd,     bias=config.bias)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project and split into Q, K, V — each (B, T, n_embd)
        q, k, v = self.qkv(x).split(C, dim=-1)

        # Reshape to (B, n_head, T, head_dim) for batched attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Fused scaled dot-product attention with causal mask.
        # is_causal=True tells SDPA to apply the causal mask internally —
        # no need to materialise the tril buffer ourselves.
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        # out: (B, n_head, T, head_dim)

        # Merge heads back: (B, T, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    Decoder-only transformer (GPT-style).

    Parameter count with default config:
      token_emb + head (tied):  32000 × 256  =  8.2M  (counted once)
      pos_emb:                    128 × 256  =  0.03M
      4 blocks × ~787k           =  3.1M
      ln_f:                                  =  0.0M
      Total:                                ≈ 11.3M params
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share parameters between input embedding and output projection.
        # This halves the vocab-related parameter count and is standard in GPT-2+.
        self.token_emb.weight = self.head.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Input length {T} exceeds block_size {self.config.block_size}"
        )
        tok = self.token_emb(idx)                             # (B, T, n_embd)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)                                   # (B, T, vocab_size)

    def num_params(self) -> int:
        # parameters() deduplicates tied weights by identity, so the shared
        # token_emb/head tensor is already counted exactly once.
        return sum(p.numel() for p in self.parameters())
