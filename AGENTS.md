---
name: sifaka-agent
description: Research-backed AI text improvement framework developer
---

# AGENTS.md - AI Agent Guide

**Purpose**: Quick reference for working on Sifaka
**Last Updated**: 2025-01-21

---

## Quick Start (First Session Commands)

**New to this repo? Run these 5 commands first:**

```bash
# 1. Verify you're on a feature branch (NEVER work on main)
git status && git branch

# 2. Run all quality checks
pytest --cov=sifaka --cov-report=term-missing
mypy sifaka/
ruff check .
black .

# 3. Run specific critic test to verify environment
pytest tests/critics/test_reflexion.py -v

# 4. Check for any TODOs or placeholders (should be NONE)
grep -r "TODO\|FIXME\|NotImplementedError" sifaka/ || echo "✅ No placeholders found"

# 5. Verify coverage is >80%
pytest --cov=sifaka | tail -1
```

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

**Core Architecture** (Why: Affects all critique operations):
- Add new critics to `sifaka/critics/core/` - Must follow research-backed patterns
- Modify improvement engine in `sifaka/core/engine/` - All critics depend on this
- Change research-backed critique methodologies (Reflexion, Constitutional AI, Self-Refine) - Research validity at stake
- Update public API in `sifaka/__init__.py` (improve, improve_sync functions) - Breaking changes for users

**Observability & Storage** (Why: Audit trail integrity):
- Change observability/audit trail implementation - Complete traceability required
- Modify storage backends in `sifaka/storage/` - Data persistence implications
- Change validation logic in `sifaka/validators/` - Quality control affected

**Dependencies & Config** (Why: Security and maintenance burden):
- Add/update dependencies in `pyproject.toml` - Increases attack surface
- Modify configuration management in `sifaka/core/config/` - System-wide effects
- Update `README.md` examples or API documentation - User-facing changes

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

**Detection Commands** (Run before committing):
```bash
# Check for security violations
grep -r "API_KEY\|SECRET\|PASSWORD" sifaka/ tests/ examples/ && echo "🚨 CREDENTIALS FOUND" || echo "✅ No credentials"

# Check for code quality violations
grep -r "TODO\|FIXME" sifaka/ && echo "🚨 TODO comments found" || echo "✅ No TODOs"

# Check for incomplete features
grep -r "NotImplementedError\|pass  # TODO" sifaka/ && echo "🚨 Placeholder code found" || echo "✅ No placeholders"

# Verify on feature branch
git branch --show-current | grep -E "^(main|master)$" && echo "🚨 ON MAIN BRANCH - CREATE FEATURE BRANCH" || echo "✅ On feature branch"

# Verify coverage >80%
pytest --cov=sifaka 2>&1 | grep "TOTAL" | awk '{if ($NF+0 < 80) print "🚨 COVERAGE " $NF " < 80%"; else print "✅ Coverage " $NF}'
```

---

## Common Mistakes & How to Avoid Them

### Mistake 1: Breaking Research-Backed Pattern
**Detection**: New critic doesn't follow Reflexion/Constitutional AI methodology
**Prevention**: Copy existing critic as template (reflexion.py)
**Fix**: Implement critique following published research
**Why It Matters**: Research validity and credibility depend on authentic methodologies

### Mistake 2: Incomplete Observability
**Detection**: Missing audit trail entries for operations
**Prevention**: Ensure all operations add to trace
**Fix**: Add trace entries for each improvement iteration
**Why It Matters**: Complete observability is core feature

### Mistake 3: Hardcoding LLM Provider
**Detection**: `from openai import OpenAI` in critic code
**Prevention**: Use PydanticAI provider abstraction
**Fix**: Replace direct provider imports with PydanticAI
**Why It Matters**: Provider-agnostic design is requirement

### Mistake 4: Not Exporting New Critics
**Detection**: New critic not importable from `sifaka`
**Prevention**: Add to `__init__.py` exports in both `critics/` and root
**Fix**: Add `from .my_critic import MyCritic` and update `__all__`
**Why It Matters**: Users can't use critic if not exported

### Mistake 5: Missing Docstrings
**Detection**: Functions without Args/Returns/Example sections
**Prevention**: Write docstring before implementation
**Fix**: Add complete docstring with all sections
**Why It Matters**: Docstrings are user documentation

### Mistake 6: Using `Any` Type
**Detection**: `grep -r "from typing import Any" sifaka/`
**Prevention**: Use specific Pydantic models for type safety
**Fix**: Create Pydantic model for response structure
**Why It Matters**: Type safety prevents bugs

### Mistake 7: Low Test Coverage
**Detection**: `pytest --cov=sifaka` shows coverage <80%
**Prevention**: Write tests as you code
**Fix**: Add unit tests until coverage >80%
**Why It Matters**: Untested critics will break

---

## Testing Decision Matrix

**When to Mock:**
- LLM API calls (OpenAI, Anthropic, Google, Groq) - Use mocked responses to avoid costs
- Network requests - Use mocked HTTP responses
- Storage backend I/O - Use temporary files or in-memory storage

**When to Use Real Dependencies:**
- Pydantic validation - Real validation catches schema bugs
- Critique logic - Real research methodology implementation
- Audit trail generation - Real trace building
- Improvement engine orchestration - Real workflow execution

**Example:**
```python
# ✅ GOOD - Mock LLM call
@pytest.mark.asyncio
async def test_critic_mocked(mocker):
    mocker.patch("sifaka.critics.core.reflexion.Agent.run")
    critic = ReflexionCritic()
    # Test logic without hitting real API

# ✅ GOOD - Real Pydantic validation
def test_result_validation():
    result = ImprovementResult(final_text="improved", improvement_score=0.95)
    assert result.improvement_score == 0.95  # Real validation

# ❌ BAD - Using real API in tests
async def test_improve():
    result = await improve("text", provider="openai")  # Costs money!
```

---

## Pre-Commit Validation

```bash
# 1. Tests pass with coverage
pytest --cov=sifaka --cov-report=term-missing
if [ $? -ne 0 ]; then echo "🚨 TESTS FAILED OR COVERAGE <80%"; exit 1; fi

# 2. Type checking clean
mypy sifaka/
if [ $? -ne 0 ]; then echo "🚨 TYPE ERRORS - FIX BEFORE COMMIT"; exit 1; fi

# 3. Linting clean
ruff check .
if [ $? -ne 0 ]; then echo "🚨 LINT ERRORS - FIX BEFORE COMMIT"; exit 1; fi

# 4. Formatted
black .

# 5. No TODOs or placeholders
grep -r "TODO\|FIXME\|NotImplementedError" sifaka/ && echo "🚨 REMOVE TODOs" && exit 1

# 6. No credentials
grep -r "API_KEY\|SECRET\|PASSWORD" sifaka/ tests/ examples/ && echo "🚨 CREDENTIALS FOUND" && exit 1

# All checks passed
echo "✅ All checks passed - ready to commit"
git add <files>
git commit -m "Clear message"
```

---

## Communication Preferences

Don't flatter me. I know what [AI sycophancy](https://www.seangoedecke.com/ai-sycophancy/) is and I don't want your praise. Be concise and direct. Don't use emdashes ever.

---

## Session Analysis & Continuous Improvement

**When to Analyze** (Multiple Triggers):
- During active sessions: After completing major tasks or every 30-60 minutes
- When failures occur: Immediately analyze and update rules
- Session end: Review entire session for patterns before closing
- User corrections: Any time user points out a mistake

**Identify Failures**:
- Framework violations (boundaries crossed, rules ignored)
- Repeated patterns (same mistake multiple times)
- Rules that didn't prevent failures
- User corrections (what needed fixing)

**Analyze Each Failure**:
- What rule should have prevented this?
- Why didn't it work? (too vague, wrong priority, missing detection pattern)
- What would have caught this earlier?

**Update AGENTS.md** (In Real-Time):
- Add new rules or strengthen existing rules immediately
- Add detection patterns (git commands, test patterns, code patterns)
- Include examples of violations and corrections
- Update priority if rule was underweighted
- Propose updates to user during session (don't wait until end)

**Priority Levels**:
- 🔴 **CRITICAL**: Security, credentials, production breaks → Update immediately, stop work
- 🟡 **IMPORTANT**: Framework violations, repeated patterns → Update with detection patterns, continue work
- 🟢 **RECOMMENDED**: Code quality, style issues → Update with examples, lowest priority

**Example Pattern**:
```
Failure: Committed TODO comments in production code (violated "No Partial Features" rule)
Detection: `grep -r "TODO" src/` before commit
Rule Update: Add pre-commit check pattern to Boundaries section
Priority: 🟡 IMPORTANT
Action Taken: Proposed rule update to user mid-session, updated AGENTS.md
```

**Proactive Analysis**:
- Before risky operations: Check if existing rules cover this scenario
- After 3+ similar operations: Look for pattern that should be codified
- When uncertainty arises: Document the decision-making gap

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
