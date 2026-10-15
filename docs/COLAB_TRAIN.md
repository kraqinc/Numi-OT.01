# Colab train (fix HybridCache)

If you see:

```text
cannot import name 'HybridCache' from 'transformers'
```

Your Colab has **transformers 5.x** and **peft** disagree. Fix:

```python
# 1) Force compatible stack
!pip install -q --upgrade "transformers>=4.46.0,<4.52.0" "peft>=0.13.0,<0.16.0" "accelerate>=0.34.0" "bitsandbytes>=0.43.0"

# 2) Restart runtime (Runtime → Restart session), then:
%cd /content
!git clone https://github.com/kraqinc/Numi-OT.01.git || true
%cd Numi-OT.01
!git pull origin main

!pip install -q -r requirements-lora.txt
!pip install -q -e .

# 3) Train
!python scripts/training/train_base_lora.py --config configs/training/lora_frontier.yaml
```

Check versions:

```python
import transformers, peft
print(transformers.__version__, peft.__version__)
# expect transformers 4.46–4.51, peft 0.13–0.15
```

Optional base if Hunyuan is heavy:

```bash
# edit config or:
# NUMI still uses model_id inside yaml
```
