FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMI_DEVICE=cpu \
    NUMI_CONFIG=configs/training/sft.yaml \
    NUMI_ENV=production

RUN useradd --create-home --uid 10001 numi

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt \
    && pip install --no-cache-dir -e .

COPY --chown=numi:numi . .

USER numi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/live')"

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
