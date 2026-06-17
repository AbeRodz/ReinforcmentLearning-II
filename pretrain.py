"""
pretrain.py — Stage 1: Pretrain TinyGPT on WikiText-2.

Teaches the model general English language and basic factual knowledge
before TriviaQA fine-tuning. Uses only a capped subset of tokens so
training stays tractable (~15 min/epoch on M3 Pro).

Optimisations from ClaseIV/trainer.py:
  bfloat16 autocast, non_blocking transfers, gradient accumulation,
  loss.detach() before scaling, grad clipping, zero_grad(set_to_none=True),
  rolling 10-step loss average, per-step scheduler, final accum flush.
  torch.compile for ~20% extra throughput on MPS/CUDA.

Run:
    python pretrain.py
    python pretrain.py --epochs 3 --max_tokens 500000 --batch_size 64
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WikiTextSFTDataset, load_wikitext
from model import GPTConfig, TinyGPT, load_tokenizer


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Training loop (mirrors trainer.py optimisations) ─────────────────────────

def train_epoch(model, loader, optimizer, scheduler, loss_fn,
                device, grad_accum, desc) -> float:
    model.train()
    losses: list[torch.Tensor] = []
    accum = 0

    bar = tqdm(loader, desc=desc, leave=False)
    for x, y in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)                          # bf16 natively
        B, T, C = logits.shape
        # Upcast to float32 for CrossEntropyLoss — softmax is numerically sensitive
        loss = loss_fn(logits.float().view(B * T, C), y.view(B * T))

        loss_log = loss.detach()
        (loss / grad_accum).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2)
        accum += 1

        if accum % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum = 0

        losses.append(loss_log)
        if len(losses) % 10 == 0:
            bar.set_description(
                f"{desc}  loss {torch.mean(torch.stack(losses[-10:])).item():.5f}"
            )

    if accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    return torch.mean(torch.stack(losses[-10:])).item()


def eval_epoch(model, loader, loss_fn, device, desc) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            B, T, C = logits.shape
            loss = loss_fn(logits.float().view(B * T, C), y.view(B * T))
            losses.append(loss.item())
    return sum(losses) / len(losses)


# ── Main ──────────────────────────────────────────────────────────────────────

def pretrain(
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_tokens: int = 600_000,
    grad_accum: int = 2,           # effective batch = 64
    save_dir: str = "./checkpoints",
    device: str | None = None,
) -> tuple[TinyGPT, GPTConfig]:
    device = device or get_device()
    print(f"[PRETRAIN] device: {device}")

    # ── Tokenizer & model ─────────────────────────────────────────────────────
    print("[PRETRAIN] Loading Mistral tokenizer …")
    tokenizer = load_tokenizer()

    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = TinyGPT(config).to(device)
    if device != "cpu":
        model = model.to(torch.bfloat16)
        print("[PRETRAIN] Model cast to bfloat16 — native bf16 on MPS/CUDA")

    # torch.compile is CUDA-only — MPS "early prototype" crashes (exit 143).
    raw_model = model
    if device == "cuda":
        try:
            model = torch.compile(model)
            raw_model = model._orig_mod
            print("[PRETRAIN] torch.compile enabled")
        except Exception as e:
            print(f"[PRETRAIN] torch.compile unavailable ({e}), running eager")

    total_params = raw_model.num_params()
    print(f"[PRETRAIN] TinyGPT params: {total_params:,}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"[PRETRAIN] Loading WikiText-2 (capped at {max_tokens:,} tokens) …")
    train_ds = WikiTextSFTDataset(load_wikitext("train"),       tokenizer, config.block_size, max_tokens)
    val_ds   = WikiTextSFTDataset(load_wikitext("validation"),  tokenizer, config.block_size, max_tokens // 10)
    print(f"[PRETRAIN] train={len(train_ds):,}  val={len(val_ds):,}  "
          f"→ {len(train_ds)//batch_size:,} batches/epoch")

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=pin)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * (len(train_loader) // grad_accum)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 10)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ── Loop ──────────────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        tl = train_epoch(model, train_loader, optimizer, scheduler,
                         loss_fn, device, grad_accum,
                         desc=f"Epoch {epoch}/{epochs} train")
        vl = eval_epoch(model, val_loader, loss_fn, device,
                        desc=f"Epoch {epoch}/{epochs} val")
        train_losses.append(tl)
        val_losses.append(vl)
        print(f"[PRETRAIN] Epoch {epoch:>2}  train={tl:.4f}  val={vl:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)

    state = raw_model.state_dict()
    ckpt_path = os.path.join(save_dir, "pretrain_checkpoint.pt")
    torch.save({"model_state_dict": state, "config": config,
                "train_losses": train_losses, "val_losses": val_losses}, ckpt_path)
    print(f"[PRETRAIN] Checkpoint → {ckpt_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epochs + 1), train_losses, marker="o", label="Train")
    ax.plot(range(1, epochs + 1), val_losses,   marker="s", label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Pretrain — TinyGPT on WikiText-2"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pretrain_loss.png"), dpi=150)
    plt.show()

    return model, config


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain TinyGPT on WikiText-2")
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--max_tokens",  type=int,   default=600_000,
                   help="Token budget from WikiText-2 train split")
    p.add_argument("--grad_accum",  type=int,   default=2)
    p.add_argument("--save_dir",                default="./checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pretrain(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_tokens=args.max_tokens,
        grad_accum=args.grad_accum,
        save_dir=args.save_dir,
    )
