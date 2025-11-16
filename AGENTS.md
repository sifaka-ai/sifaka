# AGENTS.md - AI Agent Guide

**Purpose**: Quick reference for working on Sifaka
**Last Updated**: 2025-11-16

---

## Quick Orientation

**Sifaka**: AI text improvement through research-backed critique with complete observability (v0.2.0-alpha)
**Stack**: Python 3.10+, PydanticAI 1.14+, provider-agnostic (OpenAI/Anthropic/Google/Groq)
**Coverage**: 85%+ test coverage, strict mypy, comprehensive examples

### Directory Structure

```
sifaka/
├── sifaka/
│   ├── core/
│   │   ├── config/          # Configuration management
│   │   └── engine/          # Improvement engine
│   ├── critics/
│   │   └── core/            # Critique implementations (Reflexion, Constitutional AI, etc.)
│   ├── storage/             # Storage backends (file, redis)
│   ├── tools/               # Utility tools
│   └── validators/          # Validation logic
├── examples/                # Usage examples
├── tests/                   # Unit + integration tests
└── pyproject.toml           # Dependencies and config
```

---

## Critical Rules

### 1. Research-Backed Critique Pattern
All critics implement research-backed techniques (Reflexion, Constitutional AI, Self-Refine).

```python
from sifaka.critics.core import BaseCritic

class MyCritic(BaseCritic):
    """Implement a specific research-backed critique technique."""

    async def critique(self, text: str, context: dict) -> CritiqueResult:
        """Apply critique to text.

        Args:
            text: Text to critique
            context: Additional context for critique

        Returns:
            CritiqueResult with feedback and improvement suggestions
        """
        # Implementation following research methodology
        pass
```

### 2. Provider-Agnostic Design
Must work with ANY LLM provider (OpenAI, Anthropic, Google, Groq).

```python
# ✅ GOOD
from sifaka import improve_sync
result = improve_sync("Text to improve", provider="anthropic", model="claude-3-5-sonnet")

# ❌ BAD
from openai import OpenAI
client = OpenAI()  # Hardcoded to OpenAI
```

### 3. Complete Observability
All improvement operations must provide full audit trails.

```python
result = improve_sync("Text to improve")
# Access complete trace
for iteration in result.trace:
    print(f"Iteration {iteration.number}: {iteration.improvement}")
```

### 4. Type Safety (Strict Mypy)
All functions require type hints, no `Any` without justification.

### 5. No Placeholders/TODOs
Production-grade code only. Complete implementations or nothing.

### 6. Complete Features Only
If you start, you finish:
- ✅ Implementation complete
- ✅ Tests (>80% coverage)
- ✅ Docstrings
- ✅ Example code
- ✅ Exported in `__init__.py`

### 7. PydanticAI for Structured Outputs
All critics and validators use PydanticAI for type-safe LLM responses.

---

## Development Workflow

### Before Starting
1. Check `git status` and `git branch`
2. Create feature branch: `git checkout -b feature/my-feature`

### During Development
1. Follow research-backed critique patterns
2. Write tests as you code (not after)
3. Run tests frequently: `pytest tests/`
4. Ensure complete observability (audit trails)

### Before Committing
```bash
pytest tests/      # Tests pass
mypy sifaka/       # Mypy clean
ruff check .       # Ruff clean
black .            # Black formatted
```

### After Completing
1. Add example to `examples/` if user-facing
2. Update README.md if API changed
3. Update docs if adding new features

---

## Common Tasks

### Add New Critic
```bash
# 1. Create critic file
touch sifaka/critics/core/my_critic.py

# 2. Implement BaseCritic interface
# - critique() method
# - Research-backed methodology
# - Complete observability

# 3. Export in sifaka/critics/__init__.py
from .core.my_critic import MyCritic
__all__ = [..., "MyCritic"]

# 4. Export in sifaka/__init__.py
from .critics import MyCritic
__all__ = [..., "MyCritic"]

# 5. Write tests
touch tests/critics/test_my_critic.py

# 6. Add example
touch examples/my_critic_example.py
```

### Add New Validator
```bash
# 1. Create validator file
touch sifaka/validators/my_validator.py

# 2. Implement validation logic with type safety

# 3. Export in sifaka/validators/__init__.py

# 4. Write tests
touch tests/validators/test_my_validator.py
```

### Run Tests
```bash
pytest tests/              # All tests
pytest tests/unit/         # Unit tests only
pytest -v                  # Verbose output
pytest --cov=sifaka        # Coverage report
```

---

## Code Quality Standards

### Docstrings
```python
async def improve(text: str, iterations: int = 3, provider: str = "openai") -> ImprovementResult:
    """Improve text through iterative critique.

    Args:
        text: The text to improve
        iterations: Number of improvement iterations (default: 3)
        provider: LLM provider to use (openai, anthropic, google, groq)

    Returns:
        ImprovementResult with final text, trace, and metrics

    Raises:
        ValueError: If text is empty
        ConfigError: If provider configuration is invalid

    Example:
        >>> result = await improve("Write about AI safety", iterations=2)
        >>> print(result.final_text)
        >>> print(f"Improved by {result.improvement_score:.2f}")
    """
```

### Formatting
- **black**: Line length 88
- **ruff**: Follow pyproject.toml config
- **mypy**: Strict mode (all functions typed)

---

## Quick Reference

### Key Files
- **core/engine/** - Improvement engine implementation
- **critics/core/** - Research-backed critique implementations
- **storage/** - Storage backend implementations
- **pyproject.toml** - Dependencies and config
- **README.md** - User documentation with examples
- **examples/** - Comprehensive usage examples
- **docs/** - Comprehensive documentation (being consolidated in future versions)
- **AGENTS.md** (this file) - Primary AI agent developer guide

### Key Patterns
- **Critics**: Research-backed critique techniques (Reflexion, Constitutional AI)
- **Validators**: Type-safe validation with PydanticAI
- **Storage**: Pluggable storage backends (file, redis)
- **Observability**: Complete audit trails for all operations

### Testing
```bash
pytest tests/              # Run all tests
pytest --cov=sifaka        # Coverage report
pytest -v                  # Verbose output
pytest tests/unit/         # Unit tests only
pytest tests/integration/  # Integration tests only
```

### Code Quality
```bash
black .                    # Format code
ruff check .               # Lint code
mypy sifaka/               # Type check
```

---

**Questions?** Check existing critics in `sifaka/critics/core/` or README.md (user docs)

**Last Updated**: 2025-11-16
