# Mejorar el modelo (después del smoke)

## Niveles

| Config | Params aprox | Uso |
|--------|--------------|-----|
| `sft_smoke.yaml` | ~muy pequeño | Cablear pipeline |
| **`sft_mid.yaml`** | intermedio | Mejor calidad demo Colab |
| `sft.yaml` | mayor | Cuando tengas GPU + más datos |

## Colab — mid (recomendado ahora)

```python
from getpass import getpass
token = getpass("PAT: ")
!git clone https://{token}@github.com/kraqinc/Numi-OT.01.git
%cd Numi-OT.01
!git pull
!pip install -q -r requirements-dev.txt
!pip install -q -e .

# Runtime → GPU si puedes; si no, CPU
!python scripts/training/train_sft.py --config configs/training/sft_mid.yaml --device cuda
# o: --device cpu

from google.colab import files
files.download("outputs/sft_mid/final.pt")
```

Serving config alineado: `configs/serving/mid.yaml`.

## Probar generate

```python
from serving.engine import NumiEngine
e = NumiEngine.from_checkpoint(
    "outputs/sft_mid/final.pt",
    config_path="configs/serving/mid.yaml",
    device="cpu",
)
print(e.generate([{"role":"user","content":"¿Qué modelo eres?"}], max_tokens=48).text)
```

## Cómo sigue mejorando de verdad

1. Más JSONL limpio (código, multi-turn, español técnico)
2. Tokenizer BPE entrenado (`scripts/tokenizer/train_tokenizer.py`)
3. Más capas / hidden solo si hay GPU y datos
4. DPO con `numi_preferences_seed.jsonl` después del SFT
5. Eval identidad + samples manuales antes de prod
