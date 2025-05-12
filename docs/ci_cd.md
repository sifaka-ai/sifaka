# Sifaka CI/CD Pipeline

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the Sifaka project.

## Overview

The Sifaka CI/CD pipeline is implemented using GitHub Actions and consists of the following components:

1. **Linting and Static Analysis**: Ensures code quality and consistency
2. **Automated Testing**: Runs the test suite and reports coverage
3. **Package Building**: Verifies that the package can be built correctly

## Workflow Configuration

The CI/CD workflow is defined in `.github/workflows/ci.yml` and runs on:
- Push to the `main` branch
- Pull requests to the `main` branch

## Components

### Linting and Static Analysis

The following tools are used for linting and static analysis:

- **Black**: Code formatter that ensures consistent code style
- **isort**: Import sorter that ensures consistent import ordering
- **autoflake**: Tool to remove unused imports and variables
- **Ruff**: Fast Python linter that catches common issues
- **mypy**: Static type checker that verifies type annotations
- **flake8**: Additional linter that catches code quality issues

Configuration files:
- `.flake8`: Configuration for flake8
- `pyproject.toml`: Configuration for Black, isort, and mypy
- `ruff.toml`: Configuration for Ruff

### Automated Testing

Tests are run using pytest with coverage reporting:

- **pytest**: Test runner
- **pytest-cov**: Coverage plugin for pytest
- **Codecov**: Service for tracking coverage over time

Configuration files:
- `.coveragerc`: Configuration for coverage reporting

### Package Building

The package is built using the standard Python build tools:

- **build**: Python package builder
- **wheel**: Wheel package format

## Running Locally

To run the CI/CD checks locally:

### Install Dependencies

```bash
pip install -e ".[dev]"
```

### Run Linting and Static Analysis

```bash
# Format code with Black
black sifaka tests

# Sort imports with isort
isort sifaka tests

# Check for unused imports and variables with autoflake
autoflake --check --recursive --remove-all-unused-imports --remove-unused-variables sifaka tests

# Remove unused imports and variables (use with caution)
autoflake --recursive --in-place --remove-all-unused-imports --remove-unused-variables sifaka tests

# Lint with Ruff
ruff check sifaka tests

# Type check with mypy
mypy sifaka

# Lint with flake8
flake8 sifaka tests
```

### Run Tests with Coverage

```bash
# Run tests with coverage
pytest --cov=sifaka

# Generate HTML coverage report
pytest --cov=sifaka --cov-report=html
```

### Build Package

```bash
# Install build dependencies
pip install build wheel

# Build package
python -m build
```

## CI/CD Metrics

The CI/CD pipeline tracks the following metrics:

- **Code Quality**: Measured by linting and static analysis tools
- **Test Coverage**: Percentage of code covered by tests
- **Build Success**: Whether the package builds successfully

## Maintenance

To maintain the CI/CD pipeline:

1. **Keep Dependencies Updated**: Regularly update the CI/CD tools in `setup.py` and `requirements-dev.txt`
2. **Review Configuration**: Periodically review the configuration files to ensure they match project needs
3. **Monitor Performance**: Watch CI/CD run times and optimize if necessary

## Future Enhancements

Planned enhancements to the CI/CD pipeline:

1. **Documentation Building**: Automatically build and publish documentation
2. **Dependency Scanning**: Add security scanning for dependencies
3. **Performance Testing**: Add performance benchmarks
4. **Release Automation**: Automate the release process
