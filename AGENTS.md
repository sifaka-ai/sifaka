---
name: sifaka-agent
description: Research-backed AI text improvement framework developer
---

# AGENTS.md - AI Agent Guide

**Purpose**: Quick reference for working on Sifaka
**Last Updated**: 2025-01-21

---

## Quick Orientation

**Sifaka**: AI text improvement through research-backed critique with complete observability (v0.2.0-alpha)
**Stack**: Python 3.10+, PydanticAI 1.14+, provider-agnostic (OpenAI/Anthropic/Google/Groq)
**Coverage**: 85%+ test coverage, strict mypy, comprehensive examples

### Directory Structure

```text
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

## Boundaries

### ✅ Always Do (No Permission Needed)
- Run tests: `pytest tests/`, `pytest --cov=sifaka`, `pytest -v`
- Format code: `black .`
- Lint code: `ruff check .`
- Type check: `mypy sifaka/` (strict mode required)
- Add unit tests for new critics in `tests/critics/`
- Add integration tests in `tests/integration/`
- Update docstrings when changing function signatures
- Export new critics in `__init__.py` files
- Add examples to `examples/` for new user-facing features
- Update audit trail documentation for new features

### ⚠️ Ask First
- Add new critics to `sifaka/critics/core/`
- Modify improvement engine in `sifaka/core/engine/`
- Change research-backed critique methodologies (Reflexion, Constitutional AI, Self-Refine)
- Add/update dependencies in `pyproject.toml`
- Modify storage backends in `sifaka/storage/`
- Change validation logic in `sifaka/validators/`
- Update public API in `sifaka/__init__.py` (improve, improve_sync functions)
- Change observability/audit trail implementation
- Modify configuration management in `sifaka/core/config/`
- Update `README.md` examples or API documentation

### 🚫 Never Touch

**CRITICAL SECURITY VIOLATION** ⚠️:
- **NEVER EVER COMMIT CREDENTIALS TO GITHUB**
- No API keys, tokens, passwords, secrets in ANY file
- No credentials in code, documentation, examples, tests, or configuration files
- Use environment variables (.env files in .gitignore) ONLY
- This is NON-NEGOTIABLE - violating this rule has serious security consequences

**Other Prohibitions**:
- `.env` files or API keys (use environment variables)
- Production deployment configurations
- Git history manipulation (no force push, interactive rebase on shared branches)
- User's `~/.claude/` configuration files
- Any files outside the `sifaka/` repository
- Test files to make them pass (fix the code, not the tests)
- Type checking strictness settings in `pyproject.toml`
- Coverage thresholds (must maintain >80%)
- Research methodology implementations without validation (must follow published research)

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
pytest tests/              # Run all tests (unit + integration)
pytest tests/unit/         # Run unit tests only (fast, mocked dependencies)
pytest tests/integration/  # Run integration tests only (end-to-end flows)
pytest -v                  # Run all tests with verbose output (shows test names and status)
pytest --cov=sifaka        # Generate coverage report (requires >80% coverage)
pytest --cov=sifaka --cov-report=term-missing  # Coverage with missing lines highlighted
pytest tests/critics/test_reflexion.py -v  # Run specific test file with verbose output
pytest -k "test_improve" -v  # Run tests matching pattern "test_improve"
pytest -q                  # Quiet mode (minimal output, only failures)
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
- **AGENTS.md** (this file) - Primary AI agent developer guide

### Key Patterns
- **Critics**: Research-backed critique techniques (Reflexion, Constitutional AI)
- **Validators**: Type-safe validation with PydanticAI
- **Storage**: Pluggable storage backends (file, redis)
- **Observability**: Complete audit trails for all operations

### Testing
```bash
pytest tests/              # Run all tests (unit + integration combined)
pytest --cov=sifaka        # Generate coverage report (requires >80% to pass)
pytest -v                  # Run with verbose output (shows individual test results)
pytest tests/unit/         # Run unit tests only (fast, mocked LLM calls)
pytest tests/integration/  # Run integration tests only (real critique flows with observability)
pytest -q                  # Quiet mode (minimal output, failures only)
```

### Code Quality
```bash
black .                    # Format code with black (line length 88, modifies files)
ruff check .               # Lint code with ruff (checks style and potential bugs)
mypy sifaka/               # Type check in strict mode (all functions must have type hints)
```

---

## Working with AI Agents

### Task Management
**TodoWrite enforcement (MANDATORY)**: For ANY task with 3+ distinct steps, use TodoWrite to track progress - even if the user doesn't request it explicitly. This ensures nothing gets forgotten and provides visibility into progress for everyone working on the project.

**Plan before executing**: For complex tasks, create a plan first. Understand requirements, identify dependencies, then execute systematically.

### Output Quality
**Full data display**: Show complete data structures, not summaries or truncations. Examples should display real, useful output (not "[truncated]" or "...").

**Debugging context**: When showing debug output, include enough detail to actually debug - full prompts, complete responses, actual data structures. Truncating output defeats the purpose.

**Verify usefulness**: Before showing output, verify it's actually helpful for the user's goal. Test that examples demonstrate real functionality, not abstractions.

### Audience & Context Recognition
**Auto-detect technical audiences**: Code examples, technical docs, developer presentations → eliminate ALL marketing language automatically. Engineering contexts get technical tone (no superlatives like "blazingly fast", "magnificent", "revolutionary").

**Recognize audience immediately**: Engineers get technical tone, no marketing language. Business audiences get value/ROI focus. Academic audiences get methodology and rigor. Adapt tone and content immediately based on context.

**Separate material types**: Code examples stay clean (no narratives or marketing). Presentation materials (openers, talking points) live in separate files. Documentation explains architecture and usage patterns.

### Quality & Testing
**Test output quality, not just functionality**: Run code AND verify the output is actually useful. Truncated or abstracted output defeats the purpose of examples. Show real data structures, not summaries.

**Verify before committing**: Run tests and verify examples work before showing output. Test both functionality and usefulness.

**Connect work to strategy**: Explicitly reference project milestones, coverage targets, and strategic priorities when completing work. Celebrate milestones when achieved.

### Workflow Patterns
**Iterate fast**: Ship → test → get feedback → fix → commit. Don't perfect upfront. Progressive refinement beats upfront perfection.

**Proactive problem solving**: Use tools like Glob to check file existence before execution. Anticipate common issues and handle them gracefully.

**Parallel execution**: Batch independent operations (multiple reads, parallel test execution) to improve efficiency.

### Communication & Feedback
**Direct feedback enables fast iteration**: Clear, immediate feedback on what's wrong enables rapid course correction. Specific, actionable requests work better than vague suggestions.

**Match user communication style**: Some users prefer speed over process formality, results over explanations. Adapt communication style accordingly while maintaining quality standards.

### Git & Commit Hygiene
**Commit hygiene**: Each meaningful change gets its own commit with clear message (what + why). This makes progress tracking and rollback easier.

**Clean git workflow**: Always check `git status` and `git branch` before operations. Use feature branches for all changes.

---

**Questions?** Check existing critics in `sifaka/critics/core/` or README.md (user docs)

**Last Updated**: 2025-01-22
