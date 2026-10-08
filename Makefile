PY ?= python
RUFF ?= ruff
MODEL ?= qwen25_05b_proxy
DEVICE ?= cpu
LIMIT ?= 50

.PHONY: prep entities frequency tokens forward smoke test lint format

prep:
	$(PY) -m data.prep

entities:
	$(PY) -m data.entities

frequency:
	$(PY) -m data.frequency

tokens:
	$(PY) -m crystal.tokens --write

forward:
	$(PY) -m run.forward --model $(MODEL) --device $(DEVICE) --limit $(LIMIT)

smoke: forward  # needs `make prep` and `make tokens` first
	$(PY) scripts/smoke.py --model $(MODEL) --device $(DEVICE) --limit $(LIMIT)

test:
	$(PY) -m pytest -q

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) check --fix .
	$(RUFF) format .
