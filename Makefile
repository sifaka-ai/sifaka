# Sifaka Development Makefile

.PHONY: install dev-install test lint format type-check security clean build docs pre-commit setup-hooks

# Install package for production
install:
	pip install .

# Install package for development
dev-install:
	pip install -e ".[dev]"

# Run all tests
test:
	pytest tests/ -v --cov=sifaka --cov-report=term-missing --cov-report=html

# Run tests quickly (no coverage)
test-fast:
	pytest tests/ -v --tb=short

# Run specific test file
test-file:
	pytest $(FILE) -v

# Lint code
lint:
	ruff check sifaka/
	ruff check tests/

# Format code
format:
	black sifaka/ tests/
	ruff format sifaka/ tests/

# Type checking
type-check:
	mypy sifaka/ --ignore-missing-imports

# Security scanning
security:
	bandit -r sifaka/ -f custom --msg-template "{abspath}:{line}: [{test_id}] {msg}"

# Run all quality checks
quality: lint type-check security

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete

# Build package
build: clean
	python -m build

# Install pre-commit hooks
setup-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Run pre-commit on all files
pre-commit:
	pre-commit run --all-files

# Generate documentation (if using sphinx)
docs:
	@echo "Documentation generation not yet configured"

# Run development server/example
example:
	python example.py

# Check package before publishing
check-package: build
	twine check dist/*

# Publish to PyPI (requires PYPI_TOKEN)
publish: build check-package
	twine upload dist/*

# Show coverage report in browser
coverage: test
	python -c "import webbrowser; webbrowser.open('htmlcov/index.html')"

# Show help
help:
	@echo "Sifaka Development Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  dev-install    Install package in development mode"
	@echo "  setup-hooks    Install pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  test          Run all tests with coverage"
	@echo "  test-fast     Run tests without coverage"
	@echo "  test-file     Run specific test file (make test-file FILE=tests/test_X.py)"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and ruff"
	@echo "  type-check    Run static type checking"
	@echo "  security      Run security scanning"
	@echo "  quality       Run all quality checks"
	@echo "  pre-commit    Run pre-commit on all files"
	@echo ""
	@echo "Build & Release:"
	@echo "  build         Build package"
	@echo "  check-package Check package before publishing"
	@echo "  publish       Publish to PyPI"
	@echo ""
	@echo "Utilities:"
	@echo "  clean         Clean build artifacts"
	@echo "  coverage      Show coverage report in browser"
	@echo "  example       Run example script"
	@echo "  help          Show this help message"
