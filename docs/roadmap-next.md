# Roadmap status

## Done in recent packs

CI, LoRA, system prompt runtime/full, Supabase schema+Edge Function, identity+DPO seeds, gradient checkpointing, API key, SSE, CORS, resume SFT, deploy guide, metrics, rate limit, request logging, concurrency slots, history trim, safetensors export, config validate, identity eval, Makefile, .env.example, model card, contributing.

## Still open (product / scale)

1. Train and host a real `final.pt` on Render
2. Flutter client → only `numi-chat` function
3. True token-stream inside `generate()` (yield per step)
4. Larger bilingual code corpus
5. GPU RunPod playbook with measured tokens/s
6. Online eval dashboard
7. Multi-GPU / FSDP training
8. Tool-calling protocol end-to-end with backend
9. Credits metering hooks for Numination billing
10. Red-team safety eval set
