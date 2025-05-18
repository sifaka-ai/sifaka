.PHONY: format lint test build ci install-dev

format:
	isort --profile black --line-length 100 sifaka tests
	black --line-length 100 sifaka tests
	autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables sifaka tests
	ruff check --fix --line-length 100 sifaka tests

lint:
	black --check --line-length 100 sifaka tests
	isort --check --profile black --line-length 100 sifaka tests
	autoflake --check --recursive --remove-all-unused-imports --remove-unused-variables sifaka tests
	ruff check --line-length 100 sifaka tests
	mypy sifaka

test:
	pytest --cov=sifaka --cov-report=term

build:
	python -m build

install-dev:
	pip install -e ".[dev,all]"
	pip install pre-commit
	pre-commit install

ci: lint test build
