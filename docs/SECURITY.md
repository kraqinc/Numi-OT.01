# Security baseline (Numi API)

| Control | Status |
|---------|--------|
| API key required in prod (`NUMI_API_KEY` ≥16) | Yes |
| Constant-time key compare | Yes |
| Client `system` role rejected (schema + strip) | Yes |
| Message max 8k, max 40 msgs | Yes |
| Body size limit | Yes (`NUMI_MAX_BODY_CHARS`) |
| Rate limit per key/IP + Retry-After | Yes |
| CORS warning if `*` in production | Yes |
| `/live` public; chat authenticated | Yes |
| No secrets in logs (request logging redacts later if needed) | Partial |

## Prod checklist

```text
NUMI_ENV=production
NUMI_API_KEY=<long random>
NUMI_CORS_ORIGINS=https://your.app
NUMI_RATE_LIMIT_RPM=60
NUMI_BACKEND=peft
NUMI_PEFT_BASE=tencent/Hunyuan-7B-Instruct
```

Never put service_role or NUMI_API_KEY in the Flutter app.
