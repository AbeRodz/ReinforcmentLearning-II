"""
grpo.py — Stage 2: GRPO fine-tuning of TinyGPT on TriviaQA.

GRPO (Group Relative Policy Optimization) is the RL algorithm used in
DeepSeek-R1.  Key insight: instead of a learned critic/value network (as in
PPO), we use the *group mean reward* as the baseline.

Algorithm per gradient step
───────────────────────────
1. Sample a batch of questions from TriviaQA.
2. For each question, generate G completions from the current policy (no grad).
3. Score each completion: reward = 1.0 if a correct answer appears in the
   output, else 0.0.
4. Compute group-relative advantage:
       A_i = (r_i − mean(r)) / (std(r) + ε)
5. Re-compute log-probabilities of the G completions under the current policy
   (WITH gradients) and the frozen SFT reference model (no grad).
6. Loss = −mean(A_i · log π_θ(completion_i | prompt))   ← policy gradient
         + β · KL(π_θ ∥ π_ref)                          ← stay close to SFT
7. Backprop and update θ.

The KL term prevents "reward hacking" — without it the model quickly collapses
to repeating whatever token pattern happens to contain the answer string.

Run:
    python grpo.py
    python grpo.py --episodes 800 --G 4 --kl_coeff 0.05
"""

from __future__ import annotations

import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TriviaQAGRPODataset, grpo_collate, load_trivia_qa
from model import GPTConfig, TinyGPT, load_tokenizer


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Reward ────────────────────────────────────────────────────────────────────

def compute_reward_exact(generated_text: str, answer_aliases: list[str]) -> float:
    """Binary: 1.0 if any answer appears as a substring, else 0.0."""
    gen = generated_text.lower().strip()
    return 1.0 if any(a.lower().strip() in gen for a in answer_aliases) else 0.0


_ARTICLES = {"a", "an", "the"}
_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

def _normalize(text: str) -> list[str]:
    text = text.lower().strip()
    text = "".join(c if c not in _PUNCT else " " for c in text)
    return [t for t in text.split() if t not in _ARTICLES]


def compute_reward_f1(generated_text: str, answer_aliases: list[str]) -> float:
    """SQuAD-style token F1: strips punctuation and articles before matching."""
    gen_tokens = _normalize(generated_text)
    if not gen_tokens:
        return 0.0
    gen_set = set(gen_tokens)
    best = 0.0
    for alias in answer_aliases:
        ref_tokens = _normalize(alias)
        if not ref_tokens:
            continue
        ref_set = set(ref_tokens)
        overlap = len(gen_set & ref_set)
        if overlap == 0:
            continue
        p = overlap / len(gen_tokens)
        r = overlap / len(ref_tokens)
        best = max(best, 2 * p * r / (p + r))
    return best


# ── Generation (no grad) ──────────────────────────────────────────────────────

@torch.no_grad()
def generate_group(
    model: TinyGPT,
    prompt_tokens: torch.Tensor,   # (prompt_len,)  — 1-D
    G: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> tuple[list[str], torch.Tensor]:
    """
    Sample G completions from `model` given `prompt_tokens`.

    Returns:
        texts     — decoded completion strings, length G
        full_ids  — token ids including prompt, shape (G, prompt_len + max_new_tokens)
    """
    model.eval()
    block_size = model.config.block_size

    # Repeat the prompt G times to form a batch
    idx = prompt_tokens.unsqueeze(0).expand(G, -1).clone().to(device)  # (G, P)

    for _ in range(max_new_tokens):
        # Only feed the last block_size tokens to respect positional embedding limit
        logits = model(idx[:, -block_size:])          # (G, T, vocab)
        logits = logits[:, -1, :] / temperature       # (G, vocab)
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)  # (G, 1)
        idx = torch.cat([idx, next_tok], dim=1)       # (G, P+step)

    # full_ids: (G, prompt_len + max_new_tokens)
    return idx   # texts decoded outside to keep this fn focused


# ── Log-probability computation (with grad) ───────────────────────────────────

def compute_log_probs(
    model: TinyGPT,
    full_ids: torch.Tensor,   # (G, seq_len)
    prompt_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token log-probs restricted to the *generated* portion.

    The model sees `full_ids[:, :-1]` as input and predicts `full_ids[:, 1:]`.
    We mask out the prompt positions and return:
        seq_lp      (G,)   — sum of log-probs over generated tokens (for PG loss)
        token_lp    (G, gen_len) — per-token log-probs (for KL divergence)
    """
    block_size = model.config.block_size
    G, total_len = full_ids.shape

    x = full_ids[:, :-1]   # (G, total_len-1)
    y = full_ids[:, 1:]    # (G, total_len-1)

    # Truncate to block_size if necessary
    if x.shape[1] > block_size:
        x = x[:, -block_size:]
        y = y[:, -block_size:]
        # How many prompt tokens survive after truncation?
        dropped = (total_len - 1) - block_size
        effective_prompt_end = max(0, prompt_len - 1 - dropped)
    else:
        # In (x, y) shifted-by-one space, the first generated token appears at
        # position prompt_len-1 (because y[prompt_len-1] = first generated id).
        effective_prompt_end = prompt_len - 1

    logits = model(x)                                          # (G, T, vocab)
    all_log_probs = F.log_softmax(logits, dim=-1)              # (G, T, vocab)
    token_lp = all_log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)  # (G, T)

    # Zero out prompt positions; keep only generated tokens
    gen_len = token_lp.shape[1] - effective_prompt_end
    if gen_len <= 0:
        zeros = torch.zeros(G, device=full_ids.device)
        return zeros, zeros.unsqueeze(-1)

    gen_token_lp = token_lp[:, effective_prompt_end:]   # (G, gen_len)
    seq_lp = gen_token_lp.sum(dim=-1)                   # (G,)

    return seq_lp, gen_token_lp


# ── Moving average helper ─────────────────────────────────────────────────────

def moving_average(values: list[float], window: int) -> np.ndarray:
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ── Convergence plot ──────────────────────────────────────────────────────────

def plot_convergence(
    rewards: list[float],
    losses: list[float],
    save_dir: str,
    sft_baseline: float | None = None,
) -> None:
    window = max(1, min(30, len(rewards) // 8))
    ma_r = moving_average(rewards, window)
    ma_l = moving_average(losses, window)
    x_ma = range(window - 1, len(rewards))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Reward plot ──────────────────────────────────────────────────────────
    ax1.plot(rewards, alpha=0.25, color="steelblue", linewidth=0.8, label="Raw reward")
    ax1.plot(x_ma, ma_r, color="steelblue", linewidth=2.0, label=f"MA({window})")
    if sft_baseline is not None:
        ax1.axhline(
            sft_baseline, color="orange", linestyle="--", linewidth=1.5,
            label=f"SFT baseline ({sft_baseline:.3f})",
        )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("GRPO Convergence — Reward per Episode")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Loss plot ────────────────────────────────────────────────────────────
    ax2.plot(losses, alpha=0.25, color="coral", linewidth=0.8, label="Raw loss")
    ax2.plot(x_ma, ma_l, color="coral", linewidth=2.0, label=f"MA({window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("GRPO Loss")
    ax2.set_title("GRPO Training Loss per Episode")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "grpo_convergence.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"[GRPO] Convergence plot → {path}")


# ── Main GRPO loop ────────────────────────────────────────────────────────────

def train_grpo(
    sft_checkpoint: str = "./checkpoints/sft_checkpoint.pt",
    episodes: int = 600,          # total gradient update steps
    prompts_per_step: int = 4,    # questions per gradient step
    G: int = 4,                   # completions sampled per question
    max_new_tokens: int = 32,     # generation length cap
    temperature: float = 0.8,     # sampling temperature
    lr: float = 1e-5,             # lower than SFT — fine-tuning
    kl_coeff: float = 0.1,        # weight of the KL penalty term
    reward: str = "f1",           # "exact" or "f1"
    save_dir: str = "./checkpoints",
    device: str | None = None,
) -> TinyGPT:
    device = device or get_device()
    reward_fn = compute_reward_f1 if reward == "f1" else compute_reward_exact
    print(f"[GRPO] device: {device}  reward: {reward}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("[GRPO] Loading tokenizer …")
    tokenizer = load_tokenizer()

    # ── Load SFT checkpoint ───────────────────────────────────────────────────
    print(f"[GRPO] Loading SFT checkpoint: {sft_checkpoint}")
    ckpt = torch.load(sft_checkpoint, map_location=device, weights_only=False)
    config: GPTConfig = ckpt["config"]

    policy = TinyGPT(config).to(device)
    if device != "cpu":
        policy = policy.to(torch.bfloat16)
    policy.load_state_dict(ckpt["model_state_dict"])

    # Reference model: frozen SFT copy — used only for the KL penalty.
    ref_model = copy.deepcopy(policy)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print(f"[GRPO] Policy params: {policy.num_params():,}  dtype: {next(policy.parameters()).dtype}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("[GRPO] Loading TriviaQA …")
    dataset = TriviaQAGRPODataset(
        load_trivia_qa("train", max_samples=3000), tokenizer, config.block_size
    )
    loader = DataLoader(
        dataset,
        batch_size=prompts_per_step,
        shuffle=True,
        num_workers=0,
        collate_fn=grpo_collate,
    )
    loader_iter = iter(loader)
    print(f"[GRPO] Dataset size: {len(dataset):,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = AdamW(policy.parameters(), lr=lr, weight_decay=0.01)

    os.makedirs(save_dir, exist_ok=True)

    rewards_history: list[float] = []
    loss_history: list[float] = []

    # ── Episode loop ──────────────────────────────────────────────────────────
    bar = tqdm(range(1, episodes + 1), desc="GRPO")

    for episode in bar:
        # Fetch next batch of prompts (cycle through dataset)
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        episode_rewards: list[float] = []
        step_losses: list[torch.Tensor] = []

        for item in batch:
            prompt_tokens: torch.Tensor = item["prompt_tokens"]  # (P,)
            prompt_len = prompt_tokens.shape[0]

            # ── 1. Generate G completions from current policy ────────────────
            full_ids = generate_group(
                model=policy,
                prompt_tokens=prompt_tokens,
                G=G,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )  # (G, P + max_new_tokens)

            # ── 2. Decode completions and compute rewards ────────────────────
            texts = [
                tokenizer.decode(
                    full_ids[i, prompt_len:].tolist(),
                    skip_special_tokens=True,
                )
                for i in range(G)
            ]
            rewards = torch.tensor(
                [reward_fn(t, item["answers"]) for t in texts],
                dtype=torch.float32,
                device=device,
            )  # (G,)
            episode_rewards.extend(rewards.tolist())

            # ── 3. Group-relative advantages ─────────────────────────────────
            # If all G completions have the same reward, std=0 → no signal.
            # We still do a KL update to avoid drifting from the reference.
            r_std = rewards.std()
            if r_std < 1e-8:
                advantages = torch.zeros_like(rewards)
            else:
                advantages = (rewards - rewards.mean()) / (r_std + 1e-8)  # (G,)

            # ── 4. Policy log-probs (with grad) ──────────────────────────────
            policy.train()
            seq_lp, token_lp = compute_log_probs(policy, full_ids, prompt_len)
            # seq_lp:   (G,)       — sum of log-probs per completion
            # token_lp: (G, gen_len)

            # ── 5. Reference log-probs (no grad) ─────────────────────────────
            with torch.no_grad():
                _, ref_token_lp = compute_log_probs(ref_model, full_ids, prompt_len)

            # ── 6. GRPO loss ──────────────────────────────────────────────────
            # advantages is float32; seq_lp/token_lp are bf16 — upcast to float32
            # so arithmetic doesn't error on mixed dtypes.
            pg_loss = -(advantages * seq_lp.float()).mean()

            min_len = min(token_lp.shape[-1], ref_token_lp.shape[-1])
            kl = (token_lp[..., :min_len].float() - ref_token_lp[..., :min_len].float()).mean()

            loss = pg_loss + kl_coeff * kl
            step_losses.append(loss)

        # ── 7. Aggregate losses across prompts in this step and update ────────
        if not step_losses:
            continue

        total_loss = torch.stack(step_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        rewards_history.append(mean_reward)
        loss_history.append(total_loss.item())

        bar.set_postfix(reward=f"{mean_reward:.3f}", loss=f"{total_loss.item():.4f}")

        # Periodic checkpoint every 100 episodes
        if episode % 100 == 0:
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "config": config,
                    "episode": episode,
                    "rewards_history": rewards_history,
                    "loss_history": loss_history,
                },
                os.path.join(save_dir, f"grpo_ep{episode}.pt"),
            )

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final_path = os.path.join(save_dir, "grpo_final.pt")
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "config": config,
            "episode": episodes,
            "rewards_history": rewards_history,
            "loss_history": loss_history,
        },
        final_path,
    )
    print(f"[GRPO] Final checkpoint → {final_path}")

    # ── Convergence plot ──────────────────────────────────────────────────────
    plot_convergence(rewards_history, loss_history, save_dir)

    return policy


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO stage for TinyGPT")
    p.add_argument("--sft_checkpoint", type=str, default="./checkpoints/sft_checkpoint.pt")
    p.add_argument("--episodes", type=int, default=600,
                   help="Number of gradient update steps")
    p.add_argument("--prompts_per_step", type=int, default=4,
                   help="Questions sampled per gradient step")
    p.add_argument("--G", type=int, default=4,
                   help="Completions generated per question")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl_coeff", type=float, default=0.1,
                   help="Weight of the KL penalty (higher = stay closer to SFT)")
    p.add_argument("--reward", type=str, default="f1", choices=["exact", "f1"],
                   help="Reward function: 'exact' (binary) or 'f1' (token-level F1)")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_grpo(
        sft_checkpoint=args.sft_checkpoint,
        episodes=args.episodes,
        prompts_per_step=args.prompts_per_step,
        G=args.G,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        lr=args.lr,
        kl_coeff=args.kl_coeff,
        reward=args.reward,
        save_dir=args.save_dir,
    )
