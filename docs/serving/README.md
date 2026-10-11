# Serving

## Local API

```bash
pip install -r requirements-dev.txt
python scripts/serve.py --host 0.0.0.0 --port 8000
```

Optional env:

| Variable | Meaning |
|----------|---------|
| `NUMI_CHECKPOINT` | Path to `final.pt` |
| `NUMI_CONFIG` | YAML config (default `configs/training/sft.yaml`) |
| `NUMI_DEVICE` | `auto` / `cpu` / `cuda` |
| `NUMI_MODEL_ID` | Model id returned by the API |

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (OpenAI-compatible)
- `POST /v1/chat` (simple wrapper)

Example:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "numi-ot-0.1",
    "messages": [{"role": "user", "content": "Hola"}],
    "max_tokens": 32
  }'
```

## Docker

```bash
docker compose up --build
```

## Production notes

- Run the API on a machine with enough RAM/GPU.
- Put Supabase / your app backend in front for auth and persistence.
- Do not run the LLM inside Supabase Edge Functions.
