---
base_model: HuggingFaceTB/SmolLM2-1.7B-Instruct
library_name: peft
tags:
  - numination
  - numi-ot
  - lora
---

# Numi OT LoRA v0.2

LoRA adapter for **Numination AI / Numi OT** on SmolLM2-1.7B-Instruct.

- Identity locked via system prompt + SFT seeds
- Not a vendor clone; product name is Numi OT
- Use with `serving.base_engine.BaseLoRAEngine`
