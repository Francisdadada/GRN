.PHONY: install test lint train eval api

install:
	pip install -e .[dev]

test:
	pytest

lint:
	ruff check .

train:
	python scripts/train.py --config configs/train_0.05.yaml

eval:
	python scripts/evaluate.py --config configs/eval.yaml

api:
	python scripts/serve_api.py
