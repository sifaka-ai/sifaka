# Developer Setup Guide

This guide shows how to set up Sifaka for development from scratch using `uv`.

## Complete Setup Process

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### 2. Clone and Enter Repository

```bash
git clone https://github.com/sifaka-ai/sifaka.git
cd sifaka
```

### 3. Create Virtual Environment

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
```

### 4. Install in Development Mode

```bash
# Install with all dev dependencies
uv pip install -e ".[dev]"

# This installs:
# - Sifaka in editable mode
# - pytest, ruff, mypy, black
# - pre-commit hooks
# - All optional dependencies
```

### 5. Install Pre-commit Hooks

```bash
pre-commit install
```

### 6. Set Up API Keys

Create a `.env` file in the project root:

```bash
# .env
ANTHROPIC_API_KEY="your-key-here"
OPENAI_API_KEY="your-key-here"
GOOGLE_API_KEY="your-key-here"
```

Or export them in your shell:

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

## Verify Installation

### 1. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sifaka

# Run specific test file
pytest tests/test_api.py
```

### 2. Run Linting

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy sifaka
```

### 3. Try an Example

```bash
cd examples
python constitutional_example.py
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Edit files - changes are immediately reflected thanks to `-e` install.

### 3. Run Tests

```bash
# Run tests for your changes
pytest tests/test_your_feature.py -v

# Run all tests before committing
pytest
```

### 4. Commit Changes

```bash
git add .
git commit -m "Add your feature"
# Pre-commit hooks will run automatically
```

### 5. Debug with Thought Logs

When developing, enable thought logs to see what's happening:

```python
from sifaka.storage.file import FileStorage

result = await improve(
    "test text",
    storage=FileStorage()  # Creates ./sifaka_thoughts/
)
```

## Project Structure

```
sifaka/
├── sifaka/              # Main package
│   ├── api.py          # Public API (improve, improve_sync)
│   ├── core/           # Core engine and models
│   ├── critics/        # Critic implementations
│   ├── validators/     # Validator implementations
│   └── storage/        # Storage backends
├── examples/           # Example scripts
├── tests/              # Test files
├── docs/               # Documentation
└── pyproject.toml      # Project configuration
```

## Common Development Tasks

### Adding a New Critic

1. Create file in `sifaka/critics/your_critic.py`
2. Inherit from `BaseCritic`
3. Implement required methods
4. Add tests in `tests/critics/test_your_critic.py`
5. Update documentation

### Running Examples with Different Providers

```bash
# Use Anthropic
ANTHROPIC_API_KEY="your-key" python examples/constitutional_example.py

# Use Google
GOOGLE_API_KEY="your-key" python examples/self_refine_example.py

# Use OpenAI
OPENAI_API_KEY="your-key" python examples/reflexion_example.py
```

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use FileStorage for detailed logs
from sifaka.storage.file import FileStorage
result = await improve(text, storage=FileStorage())

# Check thought logs
cat sifaka_thoughts/thoughts_*.md
```

## Troubleshooting

### Virtual Environment Issues

```bash
# Deactivate and recreate
deactivate
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Import Errors

Make sure you're in the virtual environment:

```bash
which python
# Should show: /path/to/sifaka/.venv/bin/python
```

### API Key Issues

```python
import os
print(os.getenv("ANTHROPIC_API_KEY"))  # Should not be None
```

## Tips for Development

1. **Use `uv` for all pip operations** - It's much faster than pip
2. **Always work in the virtual environment** - Check with `which python`
3. **Run pre-commit before pushing** - `pre-commit run --all-files`
4. **Use FileStorage for debugging** - See exactly what critics are doing
5. **Test with different providers** - Each has different strengths

## Next Steps

1. Read the architecture docs: `docs/architecture.md`
2. Try modifying an example in `examples/`
3. Run the test suite: `pytest -v`
4. Create your own critic or validator
