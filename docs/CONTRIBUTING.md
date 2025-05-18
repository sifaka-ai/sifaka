# Contributing to Sifaka

Thank you for your interest in contributing to Sifaka! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Documentation](#documentation)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all contributors. By participating in this project, you agree to abide by our Code of Conduct.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up your development environment (see below)
4. Create a new branch for your changes
5. Make your changes
6. Run tests to ensure your changes don't break existing functionality
7. Submit a pull request

## Development Environment

### Prerequisites

- Python 3.11 or newer
- pip
- virtualenv or conda (recommended)

### Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Coding Standards

We follow these coding standards:

- **PEP 8**: For general Python code style
- **Type Hints**: All functions and methods should include type hints
- **Docstrings**: All modules, classes, and functions should have docstrings (see [Docstring Standards](./DOCSTRING_STANDARDS.md))
- **Line Length**: Maximum line length is 100 characters
- **Imports**: Organize imports in the following order:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library specific imports
- **Naming Conventions**:
  - Classes: `CamelCase`
  - Functions/Methods: `snake_case`
  - Variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`

We use the following tools to enforce these standards:

- **black**: For code formatting
- **isort**: For import sorting
- **mypy**: For type checking
- **flake8**: For linting
- **pylint**: For additional linting

## Documentation

Good documentation is crucial for the project. Please follow these guidelines:

- Update the README.md if your changes affect the usage or installation process
- Add or update docstrings for all modules, classes, and functions
- Follow the [Docstring Standards](./DOCSTRING_STANDARDS.md)
- Update or add examples if your changes affect the API
- Consider adding or updating tutorials for significant features

## Testing

All code contributions should include tests. We use pytest for testing.

- Write unit tests for new features or bug fixes
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage (at least 80%)

To run tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=sifaka
```

## Pull Request Process

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature X that does Y"
   ```

3. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Submit a pull request to the main repository
   - Provide a clear description of the changes
   - Link any related issues
   - Explain the motivation for the changes
   - Include any necessary documentation updates

5. Address any feedback from reviewers

6. Once approved, your pull request will be merged by a maintainer

## Release Process

Releases are managed by the core team. The process typically involves:

1. Updating the version number in relevant files
2. Creating a changelog entry
3. Creating a release tag
4. Building and publishing the package to PyPI

## Questions?

If you have any questions or need help, please:

- Open an issue on GitHub
- Reach out to the maintainers

Thank you for contributing to Sifaka!
