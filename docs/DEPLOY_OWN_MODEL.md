# Deploy Numi propio — base potente independiente

```text
Flutter → Supabase → Render PEFT
  base: tencent/Hunyuan-7B-Instruct
  adapter: tu LoRA Numi
  auth: NUMI_API_KEY (Termux)
```

Sin OpenAI, sin Microsoft, sin Qwen, sin Anthropic.

## Env Render

| Key | Value |
|-----|--------|
| `NUMI_BACKEND` | `peft` |
| `NUMI_API_KEY` | tu key Termux |
| `NUMI_PEFT_BASE` | `tencent/Hunyuan-7B-Instruct` |
| `NUMI_PEFT_ADAPTER` | HF repo del adapter o `/app/adapters/final` |

7B necesita **GPU o mucha RAM**. En CPU free de Render irá muy lento o se caerá por OOM.
