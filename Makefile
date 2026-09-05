# AEGIS -- reproduction targets.
#
#   make install     editable install with dev extras
#   make test        pytest suite
#   make test-bare   same suite with no third-party test runner
#   make smoke       ~2 minute end-to-end check
#   make reproduce   full reproduction, then paper assets
#   make assets      regenerate LaTeX tables / figure data from results/
#   make lint        ruff + mypy
#   make docker      build the pinned image
#   make docker-smoke  run the smoke target inside the image
#   make clean       remove caches and generated results

PY ?= python3
PIP ?= $(PY) -m pip
IMAGE ?= aegis-redaction:1.0.0

.DEFAULT_GOAL := help
.PHONY: help install test test-bare smoke reproduce assets lint typecheck \
        docker docker-smoke docker-shell clean data-check

help:
	@grep -E '^#   ' Makefile | sed 's/^#   //'

install:
	$(PIP) install -e ".[dev,figures]"

data-check:
	$(PY) scripts/verify_data.py

test:
	$(PY) -m pytest

test-bare:
	$(PY) scripts/run_tests_nodeps.py

smoke: data-check
	cd experiments && $(PY) run_all.py --quick
	cd experiments && $(PY) make_paper_assets.py

reproduce: data-check
	cd experiments && $(PY) run_all.py
	cd experiments && $(PY) make_paper_assets.py

assets:
	cd experiments && $(PY) make_paper_assets.py

lint:
	$(PY) -m ruff check src tests experiments scripts
	$(PY) -m ruff format --check src tests experiments scripts

typecheck:
	$(PY) -m mypy

docker:
	docker build -f docker/Dockerfile -t $(IMAGE) .

docker-smoke: docker
	docker run --rm -v "$(PWD)/results:/app/results" $(IMAGE) make smoke

docker-shell: docker
	docker run --rm -it -v "$(PWD)/results:/app/results" $(IMAGE) bash

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ __pycache__
	find . -name '*.pyc' -delete
	rm -rf results/paper_assets results/exp0*.json results/run_all_manifest.json
