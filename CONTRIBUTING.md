# Contributing to Sifaka

Thank you for your interest in contributing to Sifaka! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sifaka.git
   cd sifaka
   ```
3. **Add the upstream repository** as a remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/sifaka.git
   ```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- pip and virtualenv (or similar tools)

### Setting up your environment

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the package in development mode**:
   ```bash
   pip install -e ".[dev,all]"
   ```

3. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

### Running the test suite

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=sifaka --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run tests in parallel
pytest -n auto
```

### Type checking and linting

```bash
# Run mypy type checker
mypy sifaka --strict

# Run ruff linter
ruff check sifaka tests

# Run ruff formatter
ruff format sifaka tests

# Run black formatter
black sifaka tests
```

## Making Changes

### Branch naming

Create a new branch for your changes:
```bash
git checkout -b feature/your-feature-name
git checkout -b fix/issue-description
git checkout -b docs/documentation-update
```

### Commit messages

Follow these conventions for commit messages:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests liberally after the first line

Example:
```
Add support for custom validators

- Implement BaseValidator abstract class
- Add validation hooks to improve() method
- Update documentation with examples

Fixes #123
```

## Testing

### Writing tests

- All new features must include tests
- Bug fixes should include a test that reproduces the issue
- Aim for at least 85% code coverage
- Place tests in the appropriate file under `tests/`

### Test structure

```python
def test_descriptive_name():
    """Test that the feature works as expected."""
    # Arrange
    input_text = "test input"
    
    # Act
    result = improve(input_text)
    
    # Assert
    assert result.improved_text != input_text
```

### Integration tests

For features that interact with external services:
- Mock external API calls in unit tests
- Write separate integration tests that can be skipped in CI
- Use environment variables for API keys in integration tests

## Submitting Changes

### Before submitting

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run the full test suite**:
   ```bash
   pytest
   mypy sifaka --strict
   ruff check sifaka tests
   ruff format --check sifaka tests
   ```

3. **Update documentation** if needed

4. **Add or update tests** for your changes

### Pull Request Process

1. **Create a pull request** from your fork to the main repository
2. **Fill out the PR template** completely
3. **Ensure all checks pass** (tests, linting, type checking)
4. **Request review** from maintainers
5. **Address review feedback** promptly
6. **Keep your PR up to date** with the main branch

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Write a clear description of what changes you made and why
- Include screenshots for UI changes
- Link to relevant issues
- Be patient and respectful during the review process

## Style Guidelines

### Python style

We use:
- **Black** for code formatting (line length: 88)
- **Ruff** for linting
- **MyPy** for type checking (strict mode)
- **Google-style docstrings** for documentation

### Code organization

- Keep modules focused and cohesive
- Use descriptive variable and function names
- Add type hints to all function signatures
- Write docstrings for all public functions and classes

### Example

```python
from typing import Optional

def improve_text(
    text: str,
    max_iterations: int = 3,
    timeout: Optional[float] = None,
) -> ImproveResult:
    """Improve the given text using AI-powered critique.
    
    Args:
        text: The text to improve.
        max_iterations: Maximum number of improvement iterations.
        timeout: Optional timeout in seconds.
        
    Returns:
        ImproveResult containing the improved text and metadata.
        
    Raises:
        ValueError: If text is empty or max_iterations < 1.
        TimeoutError: If the operation exceeds the timeout.
    """
    if not text:
        raise ValueError("Text cannot be empty")
    # Implementation...
```

## Documentation

### Adding documentation

- Update docstrings when changing function signatures
- Add examples to docstrings for complex functions
- Update README.md for user-facing changes
- Add entries to CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/) format
- Create or update guides in `docs/` for new features

### Documentation style

- Use clear, concise language
- Include code examples where helpful
- Explain the "why" not just the "what"
- Keep documentation up to date with code changes

## Questions?

If you have questions about contributing:
1. Check existing issues and discussions
2. Ask in a new discussion
3. Reach out to maintainers

Thank you for contributing to Sifaka!