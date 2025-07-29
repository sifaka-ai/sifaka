# Sifaka Developer Setup Guide

This guide shows how to set up Sifaka for development from scratch.

## Prerequisites

### 1. Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setting Up the Development Environment

### 1. Clone the Repository

```bash
git clone https://github.com/sifaka-ai/sifaka.git
cd sifaka
```

### 2. Create a Virtual Environment with uv

```bash
uv venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 3. Install in Development Mode

```bash
uv pip install -e ".[dev]"
```

This installs:
- Sifaka in editable mode (changes to code are reflected immediately)
- All development dependencies (pytest, ruff, mypy, etc.)
- All optional dependencies

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks run before commits.

### 5. Set up API Keys

You'll need at least one API key from these providers:

```bash
# Choose one or more:
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-google-key"
```

## Basic Usage

### Your First Improvement

Create a file `first_improvement.py`:

```python
import asyncio
from sifaka import improve

async def main():
    text = "AI is changing the world. It's important to understand."

    # Simple improvement with default settings
    result = await improve(text)

    print(f"Original: {text}")
    print(f"Improved: {result.final_text}")
    print(f"Iterations: {result.iteration}")

asyncio.run(main())
```

### Using Different Critics

```python
# Try different critics for different purposes
result = await improve(
    "Your text here",
    critics=["reflexion"],      # Self-reflection and learning
    max_iterations=3
)

# Or use multiple critics
result = await improve(
    "Your text here",
    critics=["constitutional", "self_refine"],  # Ethics + quality
    max_iterations=2
)
```

### Adding Validators

```python
from sifaka.validators.composable import Validator

# Create a tweet validator
tweet_validator = (
    Validator.create("tweet")
    .length(max_length=280)
    .contains(["#", "@"], mode="any")
    .build()
)

result = await improve(
    "Check out our new product",
    validators=[tweet_validator],
    max_iterations=3
)
```

## Choosing Providers and Models

### Fast and Cheap (Google Gemini)

```python
from sifaka import Config

result = await improve(
    text,
    provider="google",
    model="gemini-1.5-flash",
    config=Config(
        critic_model="gemini-1.5-flash",
        temperature=0.7
    )
)
```

### High Quality (Anthropic Claude)

```python
result = await improve(
    text,
    provider="anthropic",
    model="claude-3-haiku-20240307",
    config=Config(
        critic_model="claude-3-haiku-20240307",
        temperature=0.6
    )
)
```

### Balanced (OpenAI)

```python
result = await improve(
    text,
    provider="openai",
    model="gpt-4o-mini",
    config=Config(
        critic_model="gpt-4o-mini",
        temperature=0.7
    )
)
```

## Available Critics

- **reflexion** - Self-reflection and iterative learning
- **constitutional** - Principle-based evaluation for safety/ethics
- **self_refine** - General quality improvements
- **n_critics** - Multiple perspective evaluation
- **self_rag** - Fact-checking with web search (requires tools)
- **meta_rewarding** - Two-stage quality evaluation
- **self_consistency** - Consensus-based improvements
- **self_taught_evaluator** - Contrasting outputs with reasoning traces
- **agent4debate** - Multi-agent competitive debate dynamics
- **style** - Style and tone transformation
- **prompt** - Custom prompt-based critics

## Debugging with Thought Logs

Enable thought logging to see what's happening:

```python
from sifaka.storage.file import FileStorage

result = await improve(
    text,
    critics=["reflexion"],
    storage=FileStorage()  # Saves detailed logs
)

# Logs are saved in ./sifaka_thoughts/
```

## Synchronous Usage

If you're not in an async environment:

```python
from sifaka import improve_sync

# No await needed
result = improve_sync(
    "Your text here",
    critics=["self_refine"],
    max_iterations=2
)

print(result.final_text)
```

## Common Patterns

### Blog Post Improvement

```python
from sifaka.validators.composable import Validator

blog_validator = (
    Validator.create("blog")
    .length(500, 1500)
    .sentences(20, 100)
    .contains(["example", "consider"], mode="any")
    .build()
)

result = await improve(
    draft,
    critics=["self_refine", "constitutional"],
    validators=[blog_validator],
    max_iterations=3
)
```

### Technical Documentation

```python
tech_validator = (
    Validator.create("tech_doc")
    .contains(["example", "parameters", "returns"], mode="all")
    .matches(r"```[\s\S]+?```", "code_blocks")
    .build()
)

result = await improve(
    doc_draft,
    critics=["self_rag", "self_consistency"],
    validators=[tech_validator],
    config=Config(
        temperature=0.3,  # Lower for accuracy
        enable_tools=True  # For fact-checking
    )
)
```

### Quick Polish

```python
# Just one iteration for quick improvements
result = await improve(
    text,
    critics=["self_refine"],
    max_iterations=1,
    provider="google",
    model="gemini-1.5-flash"  # Fast and cheap
)
```

## Next Steps

1. **Run Examples**: Try the examples in `/examples/`
2. **Read Docs**: Check `/docs/` for detailed documentation
3. **Experiment**: Try different critics and validators
4. **Monitor Costs**: Use `result.total_cost` to track spending

## Tips

- Start with `gemini-1.5-flash` for experimentation (cheapest)
- Use `claude-3-haiku-20240307` for quality improvements
- Enable `FileStorage()` to debug what's happening
- Use validators to ensure output meets requirements
- Combine multiple critics for comprehensive improvement

## Getting Help

- **Issues**: https://github.com/sifaka-ai/sifaka/issues
- **Docs**: `/docs/README.md`
- **Examples**: `/examples/`
