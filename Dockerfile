FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMI_DEVICE=cpu \
    NUMI_ENV=production \
    NUMI_BACKEND=cloud \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 10001 numi

COPY requirements-cloud.txt pyproject.toml ./

RUN pip install --no-cache-dir -r requirements-cloud.txt

COPY --chown=numi:numi . .

RUN pip install --no-cache-dir --no-deps -e .

USER numi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/live', timeout=4)"

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]