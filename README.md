# Sifaka

**AI text improvement through research-backed critique with complete observability**

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-blue)](https://github.com/sifaka-ai/sifaka)
[![Coverage](https://img.shields.io/badge/coverage-85%25+-brightgreen)](https://github.com/sifaka-ai/sifaka)

**Status**: Alpha software (v0.2.0). Functional but early-stage. Best suited for evaluation, experimentation, and development.

---

## Why Sifaka?

**The Problem:** AI-generated text often needs refinement. How do you know if AI output is good enough? How can you systematically improve it without manual review of every output?

**What Sifaka Provides:**
- **Research-Backed Improvement**: Implements peer-reviewed critique techniques (Reflexion, Constitutional AI, Self-Refine, etc.)
- **Complete Observability**: Full audit trail showing exactly how text improved
- **Iterative Refinement**: Automatic multi-round critique and improvement cycles
- **Provider-Agnostic**: Works with OpenAI, Anthropic, Google, Groq

**Use Case Example:**
Generate product descriptions for e-commerce. Sifaka:
1. Critiques initial draft for clarity, persuasiveness, SEO
2. Iteratively refines through multiple improvement cycles
3. Validates against your criteria (length, required keywords, tone)
4. Provides complete transparency into every improvement step

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sifaka-ai/sifaka
cd sifaka

# Install with uv (recommended)
uv pip install -e .

# Or with standard pip
pip install -e .
```

## Setup

Configure your LLM provider API keys using environment variables or `.env` file:

```bash
# OpenAI (default provider)
export OPENAI_API_KEY=sk-...

# Or Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Or Google
export GOOGLE_API_KEY=...

# Or Groq
export GROQ_API_KEY=...
```

---

## Quick Start

```python
from sifaka import improve
import asyncio

async def main():
    result = await improve("Write about renewable energy benefits")
    print(result.final_text)
    print(f"\nImprovement score: {result.improvement_score:.2f}")
    print(f"Iterations: {result.iteration}")

asyncio.run(main())
```

### Synchronous API

```python
from sifaka import improve_sync

result = improve_sync("Write about renewable energy benefits")
print(result.final_text)
```

---

## Core Features

### 1. Research-Backed Critics

Sifaka implements peer-reviewed critique techniques from academic research:

| Critic | Best For | Research Paper |
|--------|----------|----------------|
| **SELF_REFINE** | General improvement | [Self-Refine (2023)](https://arxiv.org/abs/2303.17651) |
| **REFLEXION** | Learning from mistakes | [Reflexion (2023)](https://arxiv.org/abs/2303.11366) |
| **CONSTITUTIONAL** | Safety & ethics | [Constitutional AI (2022)](https://arxiv.org/abs/2212.08073) |
| **SELF_CONSISTENCY** | Balanced perspectives | [Self-Consistency (2022)](https://arxiv.org/abs/2203.11171) |
| **SELF_RAG** | Fact-checking | [Self-RAG (2023)](https://arxiv.org/abs/2310.11511) |
| **META_REWARDING** | Self-evaluation | [Meta-Rewarding (2024)](https://arxiv.org/abs/2407.19594) |
| **N_CRITICS** | Multiple perspectives | [N-Critics (2023)](https://arxiv.org/abs/2310.18679) |
| **STYLE** | Tone & style | Custom implementation |

### 2. Complete Observability

```python
result = await improve("Your text")

# Access complete audit trail
for iteration in result.trace:
    print(f"Iteration {iteration.number}")
    print(f"  Critique: {iteration.critique}")
    print(f"  Improvement: {iteration.improvement}")
    print(f"  Time: {iteration.processing_time:.2f}s")
```

### 3. Provider-Agnostic Design

```python
# OpenAI
result = await improve(text, provider="openai", model="gpt-4o-mini")

# Anthropic
result = await improve(text, provider="anthropic", model="claude-3-5-sonnet")

# Google
result = await improve(text, provider="google", model="gemini-1.5-flash")

# Groq (fast inference)
result = await improve(text, provider="groq", model="llama3-8b-8192")
```

### 4. Validation & Quality Control

```python
from sifaka.validators import LengthValidator, ContentValidator

result = await improve(
    "Write a product description",
    validators=[
        LengthValidator(min_length=100, max_length=200),
        ContentValidator(required_terms=["features", "benefits"])
    ]
)
```

---

## Usage Examples

### Example 1: Basic Improvement

```python
from sifaka import improve

result = await improve("AI is important for business.")
print(result.final_text)
# Output: "Artificial intelligence transforms business operations by automating..."
```

### Example 2: Using Specific Critics

```python
from sifaka import improve
from sifaka.core.types import CriticType

# Single critic
result = await improve(
    "Explain quantum computing",
    critics=[CriticType.REFLEXION]
)

# Multiple critics
result = await improve(
    "Explain quantum computing",
    critics=[CriticType.REFLEXION, CriticType.SELF_REFINE]
)
```

### Example 3: Style Transformation

```python
from sifaka.critics.style import StyleCritic

result = await improve(
    "We offer comprehensive solutions for your needs.",
    critics=[StyleCritic(
        style_description="Casual and friendly",
        style_examples=["Hey there!", "No worries!"]
    )]
)
```

### Example 4: Fact-Checking with SELF_RAG

```python
result = await improve(
    "The Great Wall of China is visible from space.",
    critics=[CriticType.SELF_RAG]
)
# Critiques factual accuracy and suggests corrections
```

### Example 5: Safety & Ethics Check

```python
result = await improve(
    "Guide on pest control methods",
    critics=[CriticType.CONSTITUTIONAL]
)
# Evaluates against safety principles
```

### Example 6: Multiple Perspectives

```python
result = await improve(
    "Product launch announcement",
    critics=[CriticType.N_CRITICS]
)
# Gets feedback from technical expert, general audience, editor, skeptic perspectives
```

### Example 7: Iteration Control

```python
# More iterations for higher quality
result = await improve(
    "Draft email to client",
    max_iterations=5  # Default is 3
)

# Force improvements even if validation passes
result = await improve(
    "Good text that passes validation",
    force_improvements=True
)
```

### Example 8: Configuration

```python
from sifaka import Config

config = Config(
    model="gpt-4",
    temperature=0.7,
    max_iterations=5,
    timeout_seconds=120
)

result = await improve("Your text", config=config)
```

### Example 9: Storage Backends

```python
from sifaka.storage.file import FileStorage
from sifaka.storage.redis import RedisStorage

# File storage
result = await improve(
    "Your text",
    storage=FileStorage("./results")
)

# Redis storage
result = await improve(
    "Your text",
    storage=RedisStorage("redis://localhost:6379")
)
```

### Example 10: Error Handling

```python
from sifaka.core.exceptions import ValidationError, CriticError

try:
    result = await improve(text)
except ValidationError as e:
    print(f"Validation failed: {e}")
except CriticError as e:
    print(f"Critic error: {e}")
```

### Example 11: Batch Processing

```python
import asyncio

texts = ["Text 1", "Text 2", "Text 3"]
tasks = [improve(text) for text in texts]
results = await asyncio.gather(*tasks)
```

### Example 12: Custom Validators

```python
from sifaka.validators import BaseValidator

class CustomValidator(BaseValidator):
    async def validate(self, text: str) -> ValidationResult:
        # Your custom validation logic
        passed = "important_keyword" in text.lower()
        return ValidationResult(
            validator="custom",
            passed=passed,
            message="Must contain 'important_keyword'"
        )

result = await improve(text, validators=[CustomValidator()])
```

### Example 13: Combining Critics for Comprehensive Review

```python
# Technical accuracy + readability
result = await improve(
    "Technical documentation",
    critics=[CriticType.REFLEXION, CriticType.STYLE]
)

# Safety + factual accuracy
result = await improve(
    "Health advice article",
    critics=[CriticType.CONSTITUTIONAL, CriticType.SELF_RAG]
)

# Comprehensive review
result = await improve(
    "Important business document",
    critics=[
        CriticType.SELF_REFINE,
        CriticType.N_CRITICS,
        CriticType.META_REWARDING
    ]
)
```

---

## Configuration

### Environment Variables

```bash
# LLM Provider Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...

# Optional: Default settings
SIFAKA_DEFAULT_MODEL=gpt-4o-mini
SIFAKA_MAX_ITERATIONS=3
SIFAKA_TEMPERATURE=0.7
```

### Config Object

```python
from sifaka import Config

config = Config(
    # Model settings
    model="gpt-4",              # LLM model to use
    temperature=0.7,            # Creativity (0.0-2.0)
    max_tokens=1000,            # Max response length

    # Critic settings
    critic_temperature=0.3,     # Lower = more consistent
    critic_context_window=3,    # Previous critiques to consider

    # Behavior settings
    max_iterations=3,           # Max improvement cycles
    force_improvements=False,   # Improve even if valid
    timeout_seconds=300,        # Overall timeout
)
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│           Sifaka Improvement Loop           │
└─────────────────────────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │   1. Generate/Modify     │
        │      (LLM Provider)      │
        └──────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │   2. Critique            │
        │   (Critics: Reflexion,   │
        │    Constitutional, etc)  │
        └──────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │   3. Validate            │
        │   (Validators: Length,   │
        │    Content, Custom)      │
        └──────────────────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │   4. Improve             │
        │   (Apply Suggestions)    │
        └──────────────────────────┘
                      │
                      ▼
           [Repeat or Return Result]
```

### Key Components

- **Core Engine** (`core/engine/`): Orchestrates improvement loop
- **Critics** (`critics/core/`): Research-backed critique implementations
- **Validators** (`validators/`): Quality checks and requirements
- **Storage** (`storage/`): File and Redis storage backends
- **Config** (`core/config/`): Configuration management

---

## FAQ

### General Questions

**Q: Which LLM providers are supported?**

A: OpenAI (GPT-4, GPT-3.5), Anthropic (Claude), Google (Gemini), Groq. Any OpenAI-compatible API also works.

**Q: Do I need API keys for all providers?**

A: No, only for the provider you want to use. Sifaka auto-detects available providers from environment variables.

**Q: Can I use multiple critics at once?**

A: Yes! Combine critics for comprehensive review: `critics=[CriticType.SELF_REFINE, CriticType.REFLEXION]`

**Q: How much does it cost?**

A: Costs depend on your LLM provider, model choice, text length, iterations, and critic count. Typical improvements cost $0.001-0.01 per text with efficient models (GPT-3.5-turbo, Gemini Flash).

### Critic Selection

**Q: Which critic should I use?**

- **General improvement**: `SELF_REFINE`
- **Academic/technical**: `REFLEXION` or `SELF_RAG`
- **Marketing/creative**: `STYLE`
- **Safety-critical**: `CONSTITUTIONAL`
- **Balanced perspectives**: `SELF_CONSISTENCY` or `N_CRITICS`

**Q: Can I create custom critics?**

A: Yes! Implement the `CriticPlugin` interface (see `examples/` for reference implementations).

### Performance

**Q: How can I improve performance?**

1. Use faster models: Gemini Flash or GPT-3.5-turbo
2. Reduce iterations: Set `max_iterations=1` or `2`
3. Batch processing: Process multiple texts concurrently
4. Connection pooling: Automatically enabled

**Q: Does Sifaka cache results?**

A: Not by default. Use `FileStorage` or `RedisStorage` to save results, or implement custom caching.

### Troubleshooting

**Q: Why am I getting timeout errors?**

A: Increase `timeout_seconds`, reduce `max_iterations`, or use faster models.

**Q: Why isn't my text improving?**

A: Try different temperature settings (0.7-0.9), different critics, larger models, or check input text quality.

**Q: How do I debug issues?**

A: Enable logging: `logging.basicConfig(level=logging.DEBUG)` or use Logfire integration.

### Production Use

**Q: Can I use Sifaka in production?**

A: Yes, but it's alpha software. Features: error handling, timeouts, connection pooling, storage backends, monitoring integration.

**Q: Does Sifaka work with async frameworks?**

A: Yes! Fully async, works with FastAPI, aiohttp, Django (async views), and any async Python framework.

**Q: Is there a synchronous API?**

A: Yes: `from sifaka import improve_sync`

---

## Development

For developers and contributors, see **[AGENTS.md](AGENTS.md)** for:
- Development setup and workflow
- Critical design patterns
- Code quality standards
- Testing guidelines
- Common development tasks

### Quick Commands

```bash
# Run tests
pytest tests/

# Type checking
mypy sifaka/

# Linting
ruff check .

# Formatting
black .

# Coverage
pytest --cov=sifaka
```

---

## Roadmap

### Phase 1: Core Functionality (v0.2.0) ✅
- PydanticAI 1.14+ integration
- Research-backed critics
- Provider-agnostic design
- Storage backends
- Comprehensive documentation consolidation

### Phase 2: Enhanced Critics (v0.3.0)
- More specialized critics
- Custom critic templates
- Critic performance optimization
- Enhanced validation framework

### Phase 3: Production Features (v0.4.0)
- Advanced caching strategies
- Distributed processing
- Cost optimization tools
- Performance monitoring

### Phase 4: v1.0 Release
- Production-grade stability
- Comprehensive documentation site
- Plugin ecosystem
- Enterprise features

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions welcome! This is alpha software under active development.

1. Check [GitHub Issues](https://github.com/sifaka-ai/sifaka/issues) for open tasks
2. Read [AGENTS.md](AGENTS.md) for development guidelines
3. Submit PRs with tests and documentation

---

## Research Citations

If you use Sifaka in research, please cite the underlying papers:

```bibtex
@article{madaan2023self,
  title={Self-Refine: Iterative Refinement with Self-Feedback},
  author={Madaan, Aman and others},
  journal={arXiv preprint arXiv:2303.17651},
  year={2023}
}

@article{shinn2023reflexion,
  title={Reflexion: Language Agents with Verbal Reinforcement Learning},
  author={Shinn, Noah and others},
  journal={arXiv preprint arXiv:2303.11366},
  year={2023}
}

@article{bai2022constitutional,
  title={Constitutional AI: Harmlessness from AI Feedback},
  author={Bai, Yuntao and others},
  journal={arXiv preprint arXiv:2212.08073},
  year={2022}
}
```
