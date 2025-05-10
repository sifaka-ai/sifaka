# Sifaka

A powerful and extensible framework for text validation, improvement, and critiquing.

## Features

- **Critics**: Evaluate and improve text quality
- **Adapters**: Convert between different formats
- **Rules**: Validate text against rules
- **Classifiers**: Categorize text
- **Interfaces**: Interact with the system
- **Retrieval**: Find relevant information

## Installation

```bash
# Install from PyPI
pip install sifaka

# Install from source
git clone https://github.com/yourusername/sifaka.git
cd sifaka
pip install -e .
```

## Quick Start

```python
from sifaka import Sifaka

# Initialize Sifaka
sifaka = Sifaka("config.yaml")

# Create components
critic = sifaka.create_critic("prompt_critic")
adapter = sifaka.create_adapter("pydantic_adapter")
rule = sifaka.create_rule("length_rule")

# Use components
result = critic.critique("This is a test.")
adapted = adapter.adapt(result)
is_valid = rule.validate(adapted)
```

## Command-Line Interface

```bash
# List available critics
sifaka component list critics

# Show critic configuration
sifaka component info critics prompt_critic

# Load configuration
sifaka config load config.yaml

# Update configuration
sifaka config set critics prompt_critic max_tokens 1000

# Save configuration
sifaka config save config.yaml

# Run a critic
sifaka run critic prompt_critic "This is a test."
```

## Configuration

Sifaka uses YAML configuration files. Here's an example:

```yaml
# General settings
debug: false
log_level: INFO

# Component settings
critics:
  prompt_critic:
    name: prompt_critic
    description: A critic that uses a language model to evaluate and improve text
    min_confidence: 0.7
    max_tokens: 1000
    temperature: 0.7
    cache_enabled: true
    cache_size: 100
    cache_ttl: 3600

# Cache settings
cache_enabled: true
cache_size: 1000
cache_ttl: 3600

# Performance settings
max_workers: 4
timeout: 30
```

## Components

### Critics

Critics evaluate and improve text quality:

- **PromptCritic**: Uses a language model to evaluate and improve text
- **ReflexionCritic**: Uses reflection to improve text quality
- **ConstitutionalCritic**: Evaluates text against a set of principles
- **SelfRefineCritic**: Self-refines text through multiple iterations
- **LACCritic**: Uses language-aware critique
- **SelfRAGCritic**: Uses self-retrieval augmented generation

### Adapters

Adapters convert between different formats:

- **PydanticAdapter**: Uses Pydantic for validation
- **GuardrailsAdapter**: Uses Guardrails for validation
- **ClassifierAdapter**: Adapts classifier outputs

### Rules

Rules validate text against rules:

- **LengthRule**: Checks text length
- **ContentRule**: Checks text content
- **PatternRule**: Checks text against patterns

### Classifiers

Classifiers categorize text:

- **SentimentClassifier**: Determines text sentiment
- **TopicClassifier**: Determines text topic

### Interfaces

Interfaces interact with the system:

- **RESTInterface**: REST API interface
- **CLIInterface**: Command-line interface

### Retrieval

Retrieval finds relevant information:

- **VectorRetrieval**: Vector-based retrieval system
- **KeywordRetrieval**: Keyword-based retrieval system

## Extending Sifaka

### Creating a New Critic

```python
from sifaka.critics.base import BaseCritic
from sifaka.critics.models import CriticConfig, CriticMetadata

class MyCritic(BaseCritic[str, str]):
    def __init__(self, config: CriticConfig):
        super().__init__(config)

    def validate(self, text: str) -> bool:
        return True

    def improve(self, text: str, feedback: Optional[str] = None) -> str:
        return "Improved text"

    def critique(self, text: str) -> CriticMetadata[str]:
        return CriticMetadata(
            score=0.8,
            feedback="Good text",
            issues=[],
            suggestions=[]
        )
```

### Creating a New Adapter

```python
from sifaka.adapters.base import BaseAdapter
from sifaka.adapters.models import AdapterConfig

class MyAdapter(BaseAdapter[str, str]):
    def __init__(self, config: AdapterConfig):
        super().__init__(config)

    def adapt(self, input: str) -> str:
        return "Adapted text"
```

### Creating a New Rule

```python
from sifaka.rules.base import BaseRule
from sifaka.rules.models import RuleConfig

class MyRule(BaseRule[str]):
    def __init__(self, config: RuleConfig):
        super().__init__(config)

    def validate(self, text: str) -> bool:
        return True
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/sifaka.git
cd sifaka

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_critics.py
```

### Building Documentation

```bash
# Build documentation
cd docs
make html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors
- Inspired by various text processing frameworks
- Built with modern Python practices
