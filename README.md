# Sifaka

**Simple AI text improvement through research-backed critique with complete observability**

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.1-blue)](https://github.com/sifaka-ai/sifaka)
[![CI/CD](https://github.com/sifaka-ai/sifaka/actions/workflows/ci.yml/badge.svg)](https://github.com/sifaka-ai/sifaka/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-60%25+-yellowgreen)](https://github.com/sifaka-ai/sifaka/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sifaka)](https://pypi.org/project/sifaka/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://sifaka-ai.github.io/sifaka/)


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

```python
from sifaka import improve_sync

# Simple one-liner
result = improve_sync("Climate change is bad.")
print(result.final_text)
```

📚 **[Full Documentation →](https://sifaka-ai.github.io/sifaka/)**

## Key Features

- **🔬 Research-Backed**: Implements Reflexion, Constitutional AI, Self-Refine, and more
- **👁️ Complete Observability**: Full audit trail of every improvement
- **🎯 Simple API**: One function does everything you need
- **💾 Memory-Safe**: Bounded history prevents memory leaks
- **⚡ Fast**: Minimal dependencies, maximum performance

## Documentation

### Getting Started
- **[Installation](https://sifaka-ai.github.io/sifaka/installation/)** - Installation options and setup
- **[Quickstart Guide](https://sifaka-ai.github.io/sifaka/getting-started/quickstart/)** - Get up and running in 5 minutes
- **[Basic Usage](https://sifaka-ai.github.io/sifaka/guide/basic-usage/)** - Common usage patterns
- **[API Reference](API.md)** - Complete API documentation

### User Guides
- **[Critics Guide](https://sifaka-ai.github.io/sifaka/guide/critics/)** - Available critics and usage
- **[Validators Guide](https://sifaka-ai.github.io/sifaka/guide/validators/)** - Input validation options
- **[Configuration](https://sifaka-ai.github.io/sifaka/guide/configuration/)** - Configuration options
- **[Advanced Usage](https://sifaka-ai.github.io/sifaka/guide/advanced-usage/)** - Advanced patterns

### Architecture & Development
- **[Architecture Overview](https://sifaka-ai.github.io/sifaka/architecture/)** - System design
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
- **Style** - Transform text to match specific writing styles and voices

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
