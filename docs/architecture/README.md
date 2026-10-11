# Architecture

Numi-OT.01 is a **decoder-only sparse MoE** transformer.

## Stack

| Piece | Implementation |
|-------|----------------|
| Attention | GQA + RoPE + SDPA (Flash when available) + KV-cache |
| FFN | Top-k MoE, SwiGLU experts, sorted dispatch |
| Norm | RMSNorm (pre-norm) |
| Head | Tied LM head (optional) |

## Package map

```text
architecture/
  attention/     # Attention, KVCache, RoPE helpers
  blocks/        # TransformerBlock
  embeddings/    # TokenEmbedding, LMHead
  moe/           # Router, ExpertBank, MoE, load balance
  normalization/ # RMSNorm
  positional/    # RoPE freqs
model/
  configs/       # NumiConfig (HF PretrainedConfig)
  modeling/      # NumiModel, NumiForCausalLM
```

## Config defaults

Research / smoke (`configs/training/sft.yaml`):

- layers: 4
- hidden: 512
- heads: 8 (KV heads: 2)
- experts: 4 (top-2)

Production-oriented sketch (`configs/model/` if present):

- layers: 24
- hidden: 2048
- heads: 16 (KV: 4)
- experts: 8 (top-2)

## Forward path

1. `TokenEmbedding`
2. `N × TransformerBlock` (Attention → residual → MoE → residual)
3. Final `RMSNorm`
4. `LMHead` → logits

Generation uses KV-cache: prefill full sequence, then decode one token at a time.

## Design rules

- No external model weights or architecture copy-paste as a base.
- `architecture/` must not import `model/` (avoids circular imports).
- MoE aux loss is attached during training via `aux_loss_coef`.
