FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMI_DEVICE=cpu \
    NUMI_ENV=production \
    NUMI_BACKEND=peft \
    NUMI_PEFT_BASE=HuggingFaceTB/SmolLM2-1.7B-Instruct \
    NUMI_PEFT_ADAPTER=/app/adapters/final \
    NUMI_PEFT_4BIT=0 \
    NUMI_WARMUP=1 \
    HF_HOME=/app/.cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 10001 numi \
 && mkdir -p /app/adapters /app/.cache/huggingface \
 && chown -R numi:numi /app

COPY requirements-peft.txt pyproject.toml ./

RUN pip install --no-cache-dir -r requirements-peft.txt

COPY --chown=numi:numi . .

# Bundle adapter if present in build context (outputs/lora_frontier/final)
RUN if [ -d outputs/lora_frontier/final ]; then \
      cp -a outputs/lora_frontier/final/. /app/adapters/final/; \
    fi \
 && pip install --no-cache-dir --no-deps -e . \
 && chown -R numi:numi /app

USER numi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/live', timeout=4)"

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
