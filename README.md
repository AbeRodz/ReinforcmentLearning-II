# TinyGPT-RL

DeepSeek-R1 style RLHF pipeline on a small GPT trained from scratch.  
**WikiText-2 pretrain → TriviaQA SFT → GRPO fine-tuning.**

## Model

35M parameter decoder-only transformer (GPT-style) with Mistral tokenizer (32k vocab).

## Setup

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate
```

## Run

```bash
python pretrain.py                          # Stage 1 — WikiText-2 language modeling
python sft.py                               # Stage 2 — TriviaQA supervised fine-tuning
python grpo.py --episodes 1200 --reward f1  # Stage 3 — GRPO reinforcement learning
python evaluate.py                          # Evaluate SFT vs GRPO + convergence plot
```

## Key flags

| Script | Flag | Default | Description |
|---|---|---|---|
| `pretrain.py` | `--epochs` | 1 | Training epochs |
| `pretrain.py` | `--max_tokens` | 600k | Token budget from WikiText-2 |
| `grpo.py` | `--reward` | `f1` | `f1` (SQuAD-style) or `exact` (binary) |
| `grpo.py` | `--episodes` | 600 | Gradient update steps |
| `grpo.py` | `--G` | 4 | Completions sampled per question |

## Results

| Stage | Accuracy (F1) |
|---|---|
| SFT baseline | 2.9% |
| GRPO (1200 ep, F1 reward) | 4.5% |

Checkpoints saved to `./checkpoints/`.
