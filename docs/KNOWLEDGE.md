# Conocimiento actualizado en Numi

## Por qué NO hay 100 archivos .safetensors

Los modelos “potentes” publican **muchos shards** porque son el **modelo base completo** (7B–700B+ params) partido en archivos de ~1–5 GB.

Nosotros publicamos:

| Artefacto | Qué es |
|-----------|--------|
| Base HF (SmolLM2 / Gemma / …) | Decenas de shards en el Hub del vendor |
| `numi_lora.safetensors` | **Un** adapter LoRA (~identidad Numi) |

Sumar 99 shards vacíos **no** hace el modelo más inteligente.

## Cómo se actualiza la información

1. **RAG local** — `knowledge/retrieval/simple_rag.py` + `knowledge/sources/`
2. **SFT de noticias** — `datasets/2026/numi_news_2026_10.jsonl` (reentrenar LoRA)
3. **Live tools** (producción) — Edge Function / API con búsqueda web

## Re-train con noticias

Añade el JSONL a `configs/training/lora_frontier.yaml` → `data.paths` y vuelve a correr:

```bash
python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```

## Inferencia con contexto

```python
from knowledge.retrieval.simple_rag import inject_context
from serving.base_engine import BaseLoRAEngine

q = inject_context("¿Qué pasa en Riad hoy?")
e = BaseLoRAEngine.from_adapter(...)
print(e.generate([{"role":"user","content": q}]).text)
```
