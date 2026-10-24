# Audit — Numi-OT.01 (numi-largeot branch)

Date: 2026-10-23

## Critical (fixed or must fix before usable generation)

| # | Issue | Status |
|---|--------|--------|
| 1 | README.md was corrupted (garbage text) | Fixed |
| 2 | Training used SimpleTokenizer (byte-level) with vocab_size=128256 while only ~270 ids are used → embedding waste + unreadable generation | Mitigated: train.py prefers BPE if `artifacts/tokenizer` exists; align model vocab to tokenizer |
| 3 | `bf16: true` in YAML was ignored (no AMP) | Fixed in train.py |
| 4 | LoRA applied after `model.to(device)` → adapters on CPU | Fixed earlier |
| 5 | Gradient checkpointing passed positional args to keyword-only `TransformerBlock.forward` | Fixed earlier |
| 6 | `generate()` never stops on EOS unless caller passes `eos_token_id` | Fixed: resolve from special tokens when possible |
| 7 | `lora_frontier*.yaml` still point at external bases (Qwen/DeepSeek) | Documented as legacy; own-weights path must not use them |

## High priority (next)

| # | Issue | Action |
|---|--------|--------|
| 8 | Data volume too small for ~1.3B from scratch | Scale SFT data; keep overfit tests on tiny model |
| 9 | No validation loop / early stopping | Add val split + log val loss |
| 10 | SimpleTokenizer claims large vocab_size but only maps bytes+specials | Prefer BPE; optionally clamp SimpleTokenizer.vocab_size |
| 11 | Checkpoint resume does not restore RNG / data position | Improve for exact resume |
| 12 | No automated identity/regression eval in Colab path | Wire `evaluation/suites` after train |

## Medium

| # | Issue |
|---|--------|
| 13 | Multiple parallel architecture folders (`architecture/` vs `model/modeling/`) — keep single source of truth |
| 14 | Some scripts still assume external HuggingFace tokenizers |
| 15 | Packing flag in YAML not fully implemented in ConversationDataset |
| 16 | MoE load-balance aux loss always on; document / make configurable per stage |

## Own-weights policy

- Core path: `NumiConfig` + `NumiForCausalLM` + random init + SFT/LoRA on Numi data only.
- Do not train “Numi” by LoRA-on-Qwen and call it own weights.
- Legacy frontier configs remain for experiments only; label them clearly.

## Recommended order of work

1. Train BPE → `artifacts/tokenizer`
2. SFT with that tokenizer (vocab aligned)
3. Overfit test on 10–20 examples (loss must go near zero)
4. Scale data + epochs
5. Identity + smoke eval
6. Only then scale architecture toward Large
