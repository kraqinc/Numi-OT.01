# Data

## Layout

```text
datasets/2026/
  numi_conversation_seed.jsonl   # SFT conversations
  numi_preferences_seed.jsonl    # DPO pairs
configs/data/                    # mixing / corpus configs (optional)
```

## SFT JSONL format

One object per line:

```json
{
  "messages": [
    {"role": "system", "content": "You are Numi..."},
    {"role": "user", "content": "Hola"},
    {"role": "assistant", "content": "Hola. ¿En qué te ayudo?"}
  ]
}
```

Roles supported by the chat template: `system`, `user`, `assistant`, `tool`, `tool_result`.

## DPO JSONL format

```json
{
  "prompt": "<|numi_bos|>...",
  "chosen": "good answer",
  "rejected": "bad answer"
}
```

Or `messages` + `chosen` / `rejected` (prompt is rendered from messages).

## CPT text

- Plain `.txt` (one document per line), or
- JSONL with a `text` field, or
- SFT JSONL (message contents concatenated)

## Quality guidelines

- Prefer short, correct, identity-consistent answers for seed data.
- Do not invent citations or tool results.
- Keep Spanish/English mixed examples if the product is bilingual.
