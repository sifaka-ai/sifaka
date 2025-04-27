# Contributing to Sifaka

Thank you for your interest in contributing to Sifaka! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an inclusive and respectful community.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Create a new branch** for your feature or bugfix
4. **Install development dependencies**:
   ```bash
   pip install -e ".[all,dev]"
   ```

## Development Workflow

1. **Make your changes** in your feature branch
2. **Write or update tests** to cover your changes
3. **Run the tests** to ensure they pass:
   ```bash
   pytest
   ```
4. **Format your code** using black:
   ```bash
   black sifaka tests
   ```
5. **Check your code** with flake8:
   ```bash
   flake8 sifaka tests
   ```
6. **Commit your changes** with a descriptive commit message
7. **Push your branch** to your fork on GitHub
8. **Create a pull request** to the main repository

## Pull Request Guidelines

- Fill in the required template
- Include tests for new features or bug fixes
- Update documentation if necessary
- Follow the coding style of the project
- Keep pull requests focused on a single topic

## Adding New Rules

To add a new rule to Sifaka:

1. Create a new file in the `sifaka/rules/` directory
2. Implement your rule by subclassing the `Rule` class
3. Add tests for your rule in the `tests/rules/` directory
4. Export your rule in the appropriate `__init__.py` file

Example:

```python
from sifaka.rules.base import Rule, RuleResult

class MyCustomRule(Rule):
    def __init__(self, name="my_custom_rule"):
        super().__init__(name)
    
    def validate(self, output, **kwargs):
        # Implement your validation logic here
        if some_condition:
            return RuleResult(
                passed=True,
                message="Validation passed"
            )
        else:
            return RuleResult(
                passed=False,
                message="Validation failed",
                metadata={"reason": "Some reason"}
            )
```

## Adding New Model Providers

To add a new model provider:

1. Create a new file in the `sifaka/models/` directory
2. Implement your provider by subclassing the `ModelProvider` class
3. Add tests for your provider in the `tests/models/` directory
4. Export your provider in the `sifaka/models/__init__.py` file

## License

By contributing to Sifaka, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
