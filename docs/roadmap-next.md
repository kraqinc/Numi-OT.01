# Siguientes bloques (prioridad práctica)

Ya hechos en el stack reciente: CI, LoRA, system prompt, Supabase schema/Edge Function, identity data, gradient checkpointing, API key, SSE stream, resume SFT, deploy guide.

## Próximas 10 (orden sugerido)

1. Subir un `final.pt` smoke entrenado y cablearlo en Render
2. Flutter: cliente que solo llame a `functions/v1/numi-chat`
3. Rate limit por IP / por user_id en Edge Function
4. Streaming real token-a-token en `generate()` (no solo chunk de texto)
5. Logging estructurado (request_id, latency, tokens)
6. Más corpus SFT (código + español técnico)
7. Eval automática identity ("What model are you?" → Numi OT)
8. Multi-turn history truncado por tokens, no por filas
9. Safetensors export del checkpoint
10. GPU deploy playbook (RunPod)

## No hacer

- Meter el modelo en Vercel/Netlify
- Correr PyTorch dentro de Edge Functions
- Exponer Render sin `NUMI_API_KEY`
