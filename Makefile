.PHONY: install test lint serve sft cpt dpo eval export validate docker

install:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest -q

lint:
	ruff check architecture model tokenizer training serving evaluation tests || true

serve:
	python scripts/serve.py --host 0.0.0.0 --port 8000

sft:
	python scripts/training/train_sft.py --config configs/training/sft.yaml --device cpu

cpt:
	python scripts/training/train_cpt.py --config configs/training/cpt.yaml --device cpu --max-steps 20

dpo:
	python scripts/training/train_dpo.py --config configs/training/dpo.yaml --device cpu

eval:
	python scripts/evaluation/evaluate.py --device cpu

export:
	python scripts/export_safetensors.py --input outputs/sft/final.pt --output outputs/sft/model.safetensors

validate:
	python scripts/validate_config.py --config configs/training/sft.yaml

docker:
	docker compose up --build
