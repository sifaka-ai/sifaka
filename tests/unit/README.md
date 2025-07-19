# Unit Tests

This directory contains unit tests organized by component:

- **`core/`** - Core functionality tests (config, validation, exceptions, etc.)
- **`critics/`** - Critic implementation and behavior tests
- **`validators/`** - Validator implementation and behavior tests
- **`storage/`** - Storage backend tests
- **`tools/`** - Tool integration tests
- **`plugins/`** - Plugin system tests
- **`middleware/`** - Middleware tests
- **`engine/`** - Engine component tests

## Running Unit Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run specific component tests
pytest tests/unit/critics/
pytest tests/unit/validators/

# Run with coverage
pytest tests/unit/ --cov=sifaka --cov-report=term-missing
```
