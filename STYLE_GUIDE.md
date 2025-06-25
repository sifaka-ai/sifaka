# Sifaka Style Guide

## Code Style

### Imports
- **ALWAYS** use absolute imports from package root
- **NEVER** use relative imports
- Group imports: stdlib, third-party, local

```python
# Good
from sifaka.agents.critics import ReflexionAgent
from sifaka.constants import DEFAULT_MODEL

# Bad
from ..agents.critics import ReflexionAgent
from .constants import DEFAULT_MODEL
```

### Naming
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_single_underscore`
- No double underscores except for Python magic methods

### Type Hints
- **REQUIRED** for all public functions
- Use `Optional[T]` instead of `T | None`
- Import types from `typing` module

```python
from typing import Optional, List, Dict

def improve(text: str, critics: Optional[List[str]] = None) -> str:
    ...
```

### Docstrings
- **REQUIRED** for all public functions/classes
- Use Google style docstrings
- Include examples for main API functions

```python
def improve(text: str) -> str:
    """Improve text using AI critics.

    Args:
        text: The text to improve

    Returns:
        Improved text

    Examples:
        >>> improved = improve("Hello world")
    """
```

### Error Handling
- Always raise exceptions with error codes
- Never return error objects
- Use custom exception classes

```python
from sifaka.exceptions import SifakaError
from sifaka.constants import ErrorCode

# Good
raise SifakaError("Invalid critic", code=ErrorCode.INVALID_CRITIC)

# Bad
return {"error": "Invalid critic"}
```

### Async/Sync
- Primary implementation should be async
- Always provide sync wrapper using `asyncio.run`
- Name sync versions with `_sync` suffix

```python
async def improve(text: str) -> str:
    ...

def improve_sync(text: str) -> str:
    return asyncio.run(improve(text))
```

## Architecture Principles

### Simplicity First
- Prefer simple solutions over clever ones
- Avoid premature abstraction
- Delete code that isn't being used

### Flat is Better Than Nested
- No deeply nested configurations
- No inheritance hierarchies > 2 levels
- Prefer composition over inheritance

### Explicit is Better Than Implicit
- No magic behavior
- Clear function names
- Obvious parameter names

## File Organization

```
sifaka/
├── __init__.py          # Simple public API
├── simple_api.py        # User-friendly functions
├── api.py               # Core implementation
├── constants.py         # All constants
├── config.py            # Flat configuration
├── exceptions.py        # Custom exceptions
├── agents/              # PydanticAI agents
│   ├── critics/         # Critic implementations
│   ├── models/          # Response models
│   └── tools/           # Agent tools
├── validators/          # Input validators
├── storage/             # Storage backends
└── core/                # Internal core logic
```

## Testing

### Test Organization
- Mirror source structure in tests/
- One test file per source file
- Group tests by functionality

### Test Style
- Descriptive test names
- Arrange-Act-Assert pattern
- Minimal mocking (prefer integration tests)

```python
def test_improve_with_length_constraint():
    # Arrange
    text = "Short text"
    validator = LengthValidator(min_length=50)

    # Act
    result = improve_sync(text, validators=[validator])

    # Assert
    assert len(result) >= 50
```

## Documentation

### Priority Order
1. Code should be self-documenting
2. Inline comments for complex logic only
3. Docstrings for public API
4. README for getting started
5. Detailed docs for advanced usage

### Documentation Style
- Write for beginners
- Show examples before explaining
- Keep it concise
- Update docs with code

## Pull Request Checklist

- [ ] All tests pass
- [ ] No mypy errors with --strict
- [ ] Black and ruff formatting applied
- [ ] Docstrings for new public functions
- [ ] Constants extracted (no magic strings)
- [ ] Examples added for new features
- [ ] Version bumped if needed
