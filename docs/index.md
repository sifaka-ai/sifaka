# Welcome to Sifaka

**Sifaka** is a Python library for AI-powered text improvement using research-backed critique techniques. It provides a simple API for iteratively improving text through various critique methods, with full observability and control.

## Key Features

- 🎯 **Simple API**: One function to improve any text
- 🔬 **Research-Backed**: Implements proven critique techniques from AI research
- 🔍 **Full Observability**: Complete audit trail of all improvements
- 🎨 **Highly Configurable**: Customize every aspect of the improvement process
- 🔌 **Extensible**: Easy to add custom critics and validators
- ⚡ **Async-First**: Built for performance with async/await
- 🛡️ **Type-Safe**: Full type hints and Pydantic models

## Quick Example

```python
from sifaka import improve

# Simple usage
result = await improve("Write about artificial intelligence")
print(result.improved_text)

# Advanced usage
result = await improve(
    "The Eiffel Tower is 500 meters tall.",
    critics=["reflexion", "self_rag"],
    validators=[LengthValidator(min_length=100)],
    max_iterations=3
)

# See what happened
for critique in result.critiques:
    print(f"{critique.critic}: {critique.feedback}")
```

## Installation

```bash
pip install sifaka

# With optional dependencies
pip install sifaka[anthropic]  # For Claude
pip install sifaka[tools]       # For web search tools
```

## Why Sifaka?

Named after the [Sifaka lemur](https://en.wikipedia.org/wiki/Sifaka) known for its thoughtful movements and careful decision-making, this library helps your AI carefully consider and improve its outputs through structured critique.

## Getting Started

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quickstart Guide](quickstart.md)**
  
    Get up and running in 5 minutes

- :material-book-open-variant: **[User Guide](guide/basic-usage.md)**
  
    Learn the core concepts

- :material-brain: **[Critics Guide](critics/overview.md)**
  
    Understand different critique methods

- :material-api: **[API Reference](api/core.md)**
  
    Detailed API documentation

</div>

## Popular Use Cases

- **Content Generation**: Improve blog posts, articles, and creative writing
- **Code Documentation**: Generate and refine technical documentation
- **Academic Writing**: Ensure clarity and accuracy in research
- **Business Communications**: Polish emails, reports, and proposals
- **Fact Checking**: Verify claims with web search integration

## Community

- [GitHub Repository](https://github.com/yourusername/sifaka)
- [Issue Tracker](https://github.com/yourusername/sifaka/issues)
- [Discussions](https://github.com/yourusername/sifaka/discussions)

## License

Sifaka is released under the MIT License. See the [License](about/license.md) page for details.