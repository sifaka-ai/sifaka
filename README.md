# Sifaka

**Simple AI text improvement through research-backed critique with complete observability**

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.6-blue)](https://github.com/sifaka-ai/sifaka)
[![CI/CD](https://github.com/sifaka-ai/sifaka/actions/workflows/ci.yml/badge.svg)](https://github.com/sifaka-ai/sifaka/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-60%25+-yellowgreen)](https://github.com/sifaka-ai/sifaka/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sifaka)](https://pypi.org/project/sifaka/)
[![Documentation](https://img.shields.io/badge/docs-Read%20the%20Docs-blue)](https://sifaka.readthedocs.io/)


## What is Sifaka?

Sifaka improves AI-generated text through iterative critique using research-backed techniques. Instead of hoping your AI output is good enough, Sifaka provides a transparent feedback loop where AI systems validate and improve their own outputs.

**Core Value**: See exactly how AI improves your text through research-backed techniques with complete audit trails.

## Installation

```bash
# Install from PyPI
pip install sifaka

# Or with uv
uv pip install sifaka
```

## Quick Start

### 1. Set up your API key

Sifaka requires an LLM API key. Set one of these environment variables:

```bash
export OPENAI_API_KEY="your-api-key"     # For OpenAI (GPT-4, etc.)
# or
export ANTHROPIC_API_KEY="your-api-key"  # For Claude
# or
export GEMINI_API_KEY="your-api-key"     # For Google Gemini
# or
export GROQ_API_KEY="your-api-key"       # For Groq
# or (for local Ollama - no API key needed)
export OLLAMA_BASE_URL="http://localhost:11434/v1"  # Optional, defaults to localhost
```

Or create a `.env` file in your project:
```env
OPENAI_API_KEY=your-api-key
```

**Using Ollama (Local LLMs)**:
```python
from sifaka import improve_sync, Config
from sifaka.core.config import LLMConfig

# Use Ollama with specific model (must set critic_model too!)
config = Config(
    llm=LLMConfig(
        provider="ollama",
        model="mistral:latest",
        critic_model="mistral:latest"  # Important: set this to use Ollama for critiques
    )
)
result = improve_sync("Climate change is bad.", config=config)
```

### 2. Use Sifaka

```python
from sifaka import improve_sync

# Simple one-liner
result = improve_sync("Climate change is bad.")
print(result.final_text)
```

📚 **[Full Documentation →](https://sifaka.readthedocs.io/)**

## Key Features

- **🔬 Research-Backed**: Implements Reflexion, Constitutional AI, Self-Refine, and more
- **👁️ Complete Observability**: Full audit trail of every improvement
- **🎯 Simple API**: One function does everything you need
- **💾 Memory-Safe**: Bounded history prevents memory leaks
- **⚡ Fast**: Minimal dependencies, maximum performance

## Documentation

### Getting Started
- **[Installation](https://sifaka.readthedocs.io/en/latest/installation/)** - Installation options and setup
- **[Quickstart Guide](https://sifaka.readthedocs.io/en/latest/getting-started/quickstart/)** - Get up and running in 5 minutes
- **[Basic Usage](https://sifaka.readthedocs.io/en/latest/guide/basic-usage/)** - Common usage patterns
- **[API Reference](https://sifaka.readthedocs.io/en/latest/reference/api/)** - Complete API documentation

### User Guides
- **[Critics Guide](https://sifaka.readthedocs.io/en/latest/guide/critics/)** - Available critics and usage
- **[Validators Guide](https://sifaka.readthedocs.io/en/latest/guide/validators/)** - Input validation options
- **[Configuration](https://sifaka.readthedocs.io/en/latest/guide/configuration/)** - Configuration options
- **[Advanced Usage](https://sifaka.readthedocs.io/en/latest/guide/advanced-usage/)** - Advanced patterns

### Architecture & Development
- **[Architecture Overview](https://sifaka.readthedocs.io/en/latest/architecture/)** - System design
- **[Contributing](CONTRIBUTING.md)** - How to contribute
- **[Changelog](CHANGELOG.md)** - Version history

### Examples
- **[Working Examples](examples/)** - Code examples for all features

## Research Foundation

Sifaka implements these peer-reviewed techniques:

- **[Reflexion](https://arxiv.org/abs/2303.11366)** - Self-reflection for iterative improvement
- **[Constitutional AI](https://arxiv.org/abs/2212.08073)** - Principle-based evaluation
- **[Self-Refine](https://arxiv.org/abs/2303.17651)** - Iterative self-improvement
- **[N-Critics](https://arxiv.org/abs/2310.18679)** - Ensemble of diverse perspectives
- **[Self-RAG](https://arxiv.org/abs/2310.11511)** - Retrieval-augmented critique
- **[Meta-Rewarding](https://arxiv.org/abs/2407.19594)** - Two-stage meta-evaluation
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)** - Multiple reasoning paths
- **[Self-Taught Evaluator](https://arxiv.org/abs/2408.02666)** - Contrasting outputs with reasoning traces
- **[Agent4Debate](https://arxiv.org/abs/2408.04472)** - Multi-agent competitive debate dynamics
- **Style** - Transform text to match specific writing styles and voices
- **Prompt** - Simple prompt-engineered critic for simple use cases

## Development

```bash
git clone https://github.com/sifaka-ai/sifaka
cd sifaka
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
