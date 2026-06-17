# TinyGPT RL Pipeline

RL homework: WikiText-2 pretrain → TriviaQA SFT → GRPO (DeepSeek-R1 style exact-match reward).

## Model
- 11.38M params, native bfloat16 on MPS/CUDA
- `GPTConfig`: block_size=128, n_embd=256, n_head=8, n_layer=4, dropout=0.2, vocab_size=32000
- Weight-tied token embedding + output head
- SDPA attention (`F.scaled_dot_product_attention`, `is_causal=True`)
- Mistral tokenizer (`mistralai/Mistral-7B-v0.1`, `use_fast=True`) — do not change

## Files
| File | Stage | Notes |
|---|---|---|
| `model.py` | — | TinyGPT + GPTConfig + `load_tokenizer()` |
| `dataset.py` | — | All three dataset classes; WikiText cache in `.cache/` |
| `pretrain.py` | Stage 1 | WikiText-2, 600k tokens, ~50 min on MPS |
| `sft.py` | Stage 2 | TriviaQA Q&A format, 6k train samples |
| `grpo.py` | Stage 3 | GRPO, 600 episodes, G=4 completions, binary exact-match reward |
| `evaluate.py` | Eval | SFT vs GRPO accuracy + convergence plot |

## Run order
```bash
python pretrain.py          # → checkpoints/pretrain_checkpoint.pt
python sft.py               # → checkpoints/sft_checkpoint.pt
python grpo.py              # → checkpoints/grpo_final.pt
python evaluate.py          # → checkpoints/convergence_vs_baseline.png
```

## Hardware notes
- Currently on MPS (M3 Pro 18GB); `num_workers=0` — adding workers causes unified memory pressure
- `torch.compile` disabled on MPS (crashes); auto-enabled on CUDA
- When moving to CUDA: bump `num_workers` to 4 in all DataLoaders

## Key decisions
- No autocast — model is cast to bf16 directly (`.to(torch.bfloat16)`); logits upcast to float32 before CrossEntropyLoss
- WikiText tokenization cached to `.cache/wikitext_{max_tokens}.pt` — delete cache if you change tokenizer
- GRPO reference model is a frozen deepcopy of the SFT checkpoint
- KL penalty (`kl_coeff=0.1`) prevents reward hacking
