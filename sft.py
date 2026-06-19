"""
sft.py — Stage 2: Supervised Fine-Tuning on TriviaQA Q&A format.

Loads the WikiText-2 pretrained checkpoint and fine-tunes it on
"Question: …\nAnswer: …" pairs so the model learns the chat format
before GRPO optimises for correctness.

Training loop: native bfloat16 (model weights in bf16), non_blocking
transfers, gradient accumulation, grad clipping, rolling 10-step loss
average, per-step cosine LR scheduler.

Run:
    python sft.py
    python sft.py --epochs 3 --train_samples 6000 --lr 1e-4
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

from dataset import TriviaQASFTDataset, load_trivia_qa
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

        logits = model(x)
        B, T, C = logits.shape
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

def train_sft(
    pretrain_checkpoint: str = "./checkpoints/pretrain_checkpoint.pt",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,             # lower than pretrain — fine-tuning
    train_samples: int = 6000,
    val_samples: int = 500,
    grad_accum: int = 2,
    save_dir: str = "./checkpoints",
    device: str | None = None,
) -> TinyGPT:
    device = device or get_device()
    print(f"[SFT] device: {device}")

    # ── Load pretrained model ─────────────────────────────────────────────────
    print(f"[SFT] Loading pretrain checkpoint: {pretrain_checkpoint}")
    tokenizer = load_tokenizer()
    ckpt = torch.load(pretrain_checkpoint, map_location=device, weights_only=False)
    config: GPTConfig = ckpt["config"]
    model = TinyGPT(config).to(device)
    if device != "cpu":
        model = model.to(torch.bfloat16)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[SFT] Model params: {model.num_params():,}  dtype: {next(model.parameters()).dtype}")

    # ── TriviaQA data ─────────────────────────────────────────────────────────
    print("[SFT] Loading TriviaQA …")
    train_ds = TriviaQASFTDataset(
        load_trivia_qa("train", max_samples=train_samples), tokenizer, config.block_size
    )
    val_ds = TriviaQASFTDataset(
        load_trivia_qa("validation", max_samples=val_samples), tokenizer, config.block_size
    )
    print(f"[SFT] train={len(train_ds):,}  val={len(val_ds):,}")

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=pin)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * (len(train_loader) // grad_accum)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 10)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

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
        print(f"[SFT] Epoch {epoch:>2}  train={tl:.4f}  val={vl:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "sft_checkpoint.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config,
                "train_losses": train_losses, "val_losses": val_losses}, ckpt_path)
    print(f"[SFT] Checkpoint → {ckpt_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, epochs + 1), train_losses, marker="o", label="Train")
    ax.plot(range(1, epochs + 1), val_losses,   marker="s", label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("SFT — TriviaQA Q&A Fine-Tuning"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sft_loss.png"), dpi=150)
    plt.show()

    return model


def parse_args():
    p = argparse.ArgumentParser(description="SFT on TriviaQA")
    p.add_argument("--pretrain_checkpoint", default="./checkpoints/pretrain_checkpoint.pt")
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--train_samples", type=int,   default=6000)
    p.add_argument("--val_samples",   type=int,   default=500)
    p.add_argument("--grad_accum",    type=int,   default=2)
    p.add_argument("--save_dir",                  default="./checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sft(
        pretrain_checkpoint=args.pretrain_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        grad_accum=args.grad_accum,
        save_dir=args.save_dir,
    )
