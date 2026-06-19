"""
evaluate.py — Compare SFT baseline vs GRPO fine-tuned TinyGPT.

Produces:
  • Accuracy table (SFT vs GRPO on held-out TriviaQA validation set)
  • Qualitative output examples (good and bad)
  • Convergence plot with SFT baseline line overlaid

Run after both training stages complete:
    python evaluate.py
    python evaluate.py --n_eval 300 --temperature 0.6
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import TriviaQAGRPODataset, load_trivia_qa
from grpo import compute_reward_f1 as compute_reward, generate_group, moving_average
from model import GPTConfig, TinyGPT, load_tokenizer


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Single-model eval ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: TinyGPT,
    dataset: TriviaQAGRPODataset,
    tokenizer,
    device: str,
    n_samples: int = 200,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
) -> tuple[float, list[dict]]:
    """
    Run greedy/sampled generation on `n_samples` questions and score accuracy.

    Returns:
        accuracy  — fraction of questions answered correctly
        examples  — list of dicts with question / expected / generated / correct
    """
    model.eval()
    rewards: list[float] = []
    examples: list[dict] = []

    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    for idx in indices:
        item = dataset[int(idx)]
        prompt_tokens = item["prompt_tokens"]
        answers = item["answers"]

        full_ids = generate_group(
            model=model,
            prompt_tokens=prompt_tokens,
            G=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )  # (1, P + max_new_tokens)

        prompt_len = prompt_tokens.shape[0]
        generated = tokenizer.decode(
            full_ids[0, prompt_len:].tolist(),
            skip_special_tokens=True,
        )
        reward = compute_reward(generated, item["answers"])
        rewards.append(reward)

        if len(examples) < 8:
            examples.append({
                "question":  item["question"],
                "expected":  item["answers"][0],
                "generated": generated.strip(),
                "correct":   reward > 0.0,
            })

    accuracy = float(np.mean(rewards))
    return accuracy, examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    sft_checkpoint: str = "./checkpoints/sft_checkpoint.pt",
    grpo_checkpoint: str = "./checkpoints/grpo_final.pt",
    n_eval: int = 200,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    save_dir: str = "./checkpoints",
) -> None:
    device = get_device()
    print(f"[Eval] device: {device}")

    tokenizer = load_tokenizer()

    # ── Load models ───────────────────────────────────────────────────────────
    def load_model(path: str) -> tuple[TinyGPT, GPTConfig]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg: GPTConfig = ckpt["config"]
        m = TinyGPT(cfg).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        return m, cfg

    print(f"[Eval] Loading SFT model  ← {sft_checkpoint}")
    sft_model, config = load_model(sft_checkpoint)

    print(f"[Eval] Loading GRPO model ← {grpo_checkpoint}")
    grpo_model, _ = load_model(grpo_checkpoint)

    # ── Validation dataset ────────────────────────────────────────────────────
    val_ds = TriviaQAGRPODataset(
        load_trivia_qa("validation", max_samples=500), tokenizer, config.block_size
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n[Eval] Evaluating SFT on {n_eval} examples …")
    sft_acc, sft_examples = evaluate_model(
        sft_model, val_ds, tokenizer, device, n_eval, max_new_tokens, temperature
    )

    print(f"[Eval] Evaluating GRPO on {n_eval} examples …")
    grpo_acc, grpo_examples = evaluate_model(
        grpo_model, val_ds, tokenizer, device, n_eval, max_new_tokens, temperature
    )

    # ── Results summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  SFT  accuracy : {sft_acc:.3f}  ({sft_acc*100:.1f}%)")
    print(f"  GRPO accuracy : {grpo_acc:.3f}  ({grpo_acc*100:.1f}%)")
    delta = grpo_acc - sft_acc
    print(f"  Δ improvement : {delta*100:+.1f} percentage points")
    print("=" * 55)

    # ── Qualitative examples ──────────────────────────────────────────────────
    print("\n=== GRPO Output Examples ===\n")
    for ex in grpo_examples:
        mark = "✓" if ex["correct"] else "✗"
        print(f"[{mark}] Q : {ex['question']}")
        print(f"    Expected  : {ex['expected']}")
        print(f"    Generated : {ex['generated'][:120]}")
        print()

    # ── Convergence plot with SFT baseline ───────────────────────────────────
    grpo_ckpt = torch.load(grpo_checkpoint, map_location="cpu", weights_only=False)
    rewards_history: list[float] = grpo_ckpt.get("rewards_history", [])

    if rewards_history:
        window = max(1, min(30, len(rewards_history) // 8))
        ma = moving_average(rewards_history, window)
        x_ma = range(window - 1, len(rewards_history))

        plt.figure(figsize=(10, 5))
        plt.plot(rewards_history, alpha=0.2, color="steelblue", linewidth=0.8, label="Raw reward")
        plt.plot(x_ma, ma, color="steelblue", linewidth=2.0, label=f"MA({window})")
        plt.axhline(
            sft_acc,
            color="orange",
            linestyle="--",
            linewidth=1.8,
            label=f"SFT baseline ({sft_acc:.3f})",
        )
        plt.xlabel("Episode")
        plt.ylabel("Mean Reward")
        plt.title("GRPO Reward Convergence vs SFT Baseline")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        path = os.path.join(save_dir, "convergence_vs_baseline.png")
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[Eval] Plot saved → {path}")
    else:
        print("[Eval] No rewards history found in GRPO checkpoint — skipping plot.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SFT vs GRPO TinyGPT")
    p.add_argument("--sft_checkpoint", default="./checkpoints/sft_checkpoint.pt")
    p.add_argument("--grpo_checkpoint", default="./checkpoints/grpo_final.pt")
    p.add_argument("--n_eval", type=int, default=200,
                   help="Number of validation questions to evaluate on")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--save_dir", default="./checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        sft_checkpoint=args.sft_checkpoint,
        grpo_checkpoint=args.grpo_checkpoint,
        n_eval=args.n_eval,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        save_dir=args.save_dir,
    )
