"""
dataset.py — Datasets for all three training stages.

Stage 1 — Pretrain  : WikiText-2 sliding-window next-token prediction
Stage 2 — SFT       : TriviaQA "Question: … Answer: …" next-token prediction
Stage 3 — GRPO      : TriviaQA prompt-only windows; reward = exact match
"""

from __future__ import annotations

import gc
import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer


# ── Stage 1: WikiText-2 ───────────────────────────────────────────────────────

def load_wikitext(split: str = "train") -> str:
    """Return all WikiText-2 text for a given split as one concatenated string."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return "\n".join(row["text"] for row in ds if row["text"].strip())


class WikiTextSFTDataset(Dataset):
    """
    Sliding-window next-token prediction over WikiText-2.
    Same pattern as CharDataset in ClaseIV but token-level with Mistral tokenizer.

    Tokenisation is cached to disk so restarts skip the encode() call entirely.
    Cache is keyed by split + max_tokens — different configs get different files.
    """

    def __init__(
        self,
        text: str,
        tokenizer: PreTrainedTokenizer,
        block_size: int,
        max_tokens: int = 300_000,
        cache_dir: str = ".cache",
    ) -> None:
        cache_path = os.path.join(cache_dir, f"wikitext_{max_tokens}.pt")

        if os.path.exists(cache_path):
            print(f"[Dataset] Loading tokenised cache: {cache_path}")
            self.data = torch.load(cache_path, weights_only=True)
        else:
            print("[Dataset] Tokenising WikiText-2 (first run only) …")
            ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
            self.data = torch.tensor(ids, dtype=torch.long)
            del ids; gc.collect()
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(self.data, cache_path)
            print(f"[Dataset] Cache saved → {cache_path}")

        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# ── Stage 2: TriviaQA SFT ────────────────────────────────────────────────────

def load_trivia_qa(split: str = "train", max_samples: int | None = None):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def build_qa_text(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


class TriviaQASFTDataset(Dataset):
    """
    Tokenises Q&A pairs as flat sequences for next-token prediction.
    Sequences shorter than block_size+1 are right-padded; loss ignores padding.

    Uses batch encoding so the Rust fast-tokenizer runs in parallel across all
    examples instead of one Python call per item.
    """

    def __init__(self, data, tokenizer: PreTrainedTokenizer, block_size: int) -> None:
        self.block_size = block_size
        self.pad_id = tokenizer.pad_token_id
        self.examples: list[torch.Tensor] = []

        texts = [build_qa_text(item["question"], item["answer"]["value"]) for item in data]
        encoded = tokenizer(texts, add_special_tokens=True)["input_ids"]

        for ids in encoded:
            if len(ids) < 2:
                continue
            self.examples.append(torch.tensor(ids[: block_size + 1], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        tokens = self.examples[idx]
        if len(tokens) < self.block_size + 1:
            pad = torch.full(
                (self.block_size + 1 - len(tokens),), self.pad_id, dtype=torch.long
            )
            tokens = torch.cat([tokens, pad])
        return tokens[: self.block_size], tokens[1 : self.block_size + 1]


# ── Stage 3: TriviaQA GRPO ───────────────────────────────────────────────────

class TriviaQAGRPODataset(Dataset):
    """
    Returns (prompt_tokens, answer_aliases) for GRPO.
    Prompt is capped at block_size // 2 tokens to leave room for generation.

    Batch-encodes all prompts at once — the fast tokenizer handles this in Rust.
    """

    def __init__(self, data, tokenizer: PreTrainedTokenizer, block_size: int) -> None:
        self.examples: list[dict] = []
        max_prompt = block_size // 2

        data = list(data)
        prompts = [build_prompt(item["question"]) for item in data]
        encoded = tokenizer(prompts, add_special_tokens=True)["input_ids"]

        for item, prompt_ids in zip(data, encoded):
            value = item["answer"]["value"]
            aliases = item["answer"]["aliases"]
            all_answers = list(dict.fromkeys([value] + aliases))
            self.examples.append(
                {
                    "prompt_tokens": torch.tensor(prompt_ids[:max_prompt], dtype=torch.long),
                    "answers": all_answers,
                    "question": item["question"],
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def grpo_collate(batch: list[dict]) -> list[dict]:
    return batch
