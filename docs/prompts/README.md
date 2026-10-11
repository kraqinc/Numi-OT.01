# System prompts — Numination AI / Numi OT

## Layers

| Mode | Module | Use |
|------|--------|-----|
| **Runtime** | `tokenizer.templates.system.RUNTIME_SYSTEM_PROMPT` | Injected into chat (API, SFT seeds) |
| **Full** | `FULL_SYSTEM_PROMPT` | Product/ops reference, long context |

## Improvements vs the raw extended draft

1. **Split runtime vs full** — a 2k+ token manifesto on every turn wastes context; the runtime version keeps identity + behavior tight.
2. **Identity lock** — Numi OT only; no vendor claims.
3. **Vision ≠ production** — explicit rule against treating the product map as finished.
4. **Language** — Spanish default + mirror user language.
5. **Honesty / no invention** — endpoints, status, tools, secrets.
6. **Layered diagnosis** — Flutter vs Supabase vs provider when debugging.

## Usage

```python
from tokenizer.templates.system import (
    DEFAULT_NUMI_SYSTEM_PROMPT,
    ensure_system_message,
)

messages = ensure_system_message([{"role": "user", "content": "Hola"}])
# API engine does this automatically when inject_system=True
```

## Editing policy

- Change **runtime** when behavior for the model must change.
- Change **full** when the product map / QA / state notes change.
- Keep both aligned on identity, motto, and repos.
