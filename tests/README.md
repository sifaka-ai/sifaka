# Sifaka Tests

This directory contains tests for the Sifaka framework.

## Test Structure

The test structure mirrors the package structure:

```
tests/
├── conftest.py                # Global pytest fixtures and configuration
├── test_sifaka.py             # Tests for the main package
├── chain/                     # Tests for chain components
│   ├── test_chain.py          # Tests for Chain class
│   ├── test_engine.py         # Tests for Engine class
│   └── ...
├── critics/                   # Tests for critics components
│   ├── test_prompt_critic.py  # Tests for PromptCritic
│   └── ...
├── models/                    # Tests for model providers
│   ├── test_openai.py         # Tests for OpenAIProvider
│   └── ...
├── rules/                     # Tests for rules
│   ├── test_length_rule.py    # Tests for LengthRule
│   └── ...
└── utils/                     # Tests for utilities
    ├── test_config.py         # Tests for configuration utilities
    └── ...
```

## Running Tests

To run all tests:

```bash
pytest
```

To run tests for a specific module:

```bash
pytest tests/chain/
```

To run a specific test file:

```bash
pytest tests/chain/test_chain.py
```

## Test Fixtures

Common test fixtures are defined in `conftest.py` files:

- Global fixtures in `tests/conftest.py`
- Module-specific fixtures in `tests/<module>/conftest.py`

## Mock Components

Mock components are provided for testing:

- `MockProvider` for testing without real LLM API calls
- Mock validators and improvers for testing chains
- Mock retrievers for testing retrieval components

## Test Coverage

To run tests with coverage:

```bash
pytest --cov=sifaka
```

To generate a coverage report:

```bash
pytest --cov=sifaka --cov-report=html
```
