# How Numi learns new things

Weights (LoRA) change only when you **re-train**.  
Facts and product news can be added **live** with RAG:

```text
New .md in knowledge/learn/
        ↓
retrieve() on each chat (NUMI_RAG=1)
        ↓
context injected into last user message
        ↓
model answers using that context
```

## Teach a fact (seconds)

```bash
python scripts/learn_fact.py "PayPal reload URL is https://paypal.me/YOURUSER"
python scripts/learn_fact.py --file ./changelog.md
```

## Env

| Var | Default | Meaning |
|-----|---------|--------|
| `NUMI_RAG` | `1` | Inject knowledge |
| `NUMI_KNOWLEDGE_DIRS` | — | Extra dirs `:`-separated |

## When to re-train instead

- New **style** or **skill** (better code, better Spanish)
- Behavior change permanent → add JSONL + `train_base_lora.py`

## When to only learn/

- Prices, dates, release notes, internal docs, “what we shipped today”
