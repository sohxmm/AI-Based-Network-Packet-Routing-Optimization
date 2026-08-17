# One command per thing you might want to do.
#
# `make up` exists because the reviewer's first action is `git clone && docker
# compose up`, and that used to fail immediately: three services declared
# `env_file: .env`, .env is gitignored, and nothing created it.

.PHONY: help up down logs test lint bench train report verify clean

help:
	@echo "make up      - build and start the full stack (creates .env if missing)"
	@echo "make down    - stop everything and remove volumes"
	@echo "make logs    - follow the backend logs"
	@echo "make test    - run the backend and frontend test suites"
	@echo "make lint    - ruff + eslint"
	@echo "make train   - train every model from scratch (~35 min on a laptop CPU)"
	@echo "make bench   - run every benchmark scenario"
	@echo "make report  - regenerate benchmark figures and the results document"
	@echo "make verify  - honesty gates: do the documented claims match the artifacts?"

up:
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")
	docker compose up --build

down:
	docker compose down -v

logs:
	docker compose logs -f backend

test:
	python -m pytest -m "not slow"
	cd web && npm run test

lint:
	ruff check .
	cd web && npm run lint

train:
	python -m ml.training.train_gnn
	python -m ml.training.train_rl
	python -m ml.training.train_lstm
	python -m ml.training.train_regional

bench:
	python -m experiments.runner --scenario all

report:
	python -m experiments.report

verify:
	python -m pytest tests/honesty -v
	python scripts/verify_claims.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache web/dist
