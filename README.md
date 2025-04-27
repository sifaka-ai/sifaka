# Sifaka: Reflection and Reliability for LLMs

**Sifaka** is an open-source framework that adds **reflection and reliability** to LLM applications. Build safer, more reliable AI systems that:

- ✅ Catch hallucinations before they reach users
- ✅ Enforce rules and tone consistency
- ✅ Provide transparency and auditability

[![PyPI version](https://badge.fury.io/py/sifaka.svg)](https://badge.fury.io/py/sifaka)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📚 Table of Contents
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Migration Guide](#migration-guide)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```python
# Simple validation example
from sifaka.chain import Chain
from sifaka.models import OpenAIProvider
from sifaka.rules import LengthRule, ProhibitedContentRule
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig

# Configure provider
provider = OpenAIProvider(
    model_name="gpt-4-turbo-preview",
    config={"api_key": "your-api-key"}
)

# Configure rules
rules = [
    LengthRule(name="length", min_length=50, max_length=500),
    ProhibitedContentRule(name="content_filter", prohibited_terms=["inappropriate", "offensive"])
]

# Configure critic
critic = PromptCritic(
    model=provider,
    config=PromptCriticConfig(
        name="text_improver",
        system_prompt="You are an expert editor that improves text."
    )
)

# Create chain
chain = Chain(model=provider, rules=rules, critic=critic, max_attempts=3)

# Generate content with validation and reflection
result = chain.run("Write a professional email to a colleague about a project deadline.")
print(result.content)  # Final content
print(result.passed_validation)  # True if all rules passed
```

## Core Components

Sifaka has three main components:

### 1. Chain
The orchestrator that manages the validation and improvement cycle.

```python
from sifaka.chain import Chain

chain = Chain(
    model=provider,       # LLM provider
    rules=[rule1, rule2], # Validation rules
    critic=critic,        # Optional critic for improvement
    max_attempts=3        # Max improvement attempts
)
```

### 2. Rules
Enforce constraints on content with specialized validators.

```python
from sifaka.rules import (
    # Content validation
    LengthRule, ProhibitedContentRule, FormatRule,

    # Pattern detection
    SymmetryRule, RepetitionRule,

    # Safety checks
    ToxicityRule, BiasRule, HarmfulContentRule,

    # Domain-specific
    LegalCitationRule, MedicalRule, CodeRule
)
```

### 3. Critics
Analyze and improve content based on rule failures.

```python
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig

critic = PromptCritic(
    model=provider,
    config=PromptCriticConfig(
        name="editor",
        system_prompt="You are an expert editor..."
    )
)
```

### 4. Classifiers
Analyze text characteristics without enforcing rules.

```python
from sifaka.classifiers import (
    SentimentClassifier,     # Analyze sentiment
    ReadabilityClassifier,   # Assess reading level
    ToxicityClassifier       # Detect toxic content
)

# Use directly
classifier = SentimentClassifier()
result = classifier.classify("This is amazing!")
print(result.label)  # "positive"

# Or with a rule
from sifaka.rules import ClassifierRule
sentiment_rule = ClassifierRule(classifier=SentimentClassifier())
```

## Key Features

### Validation Modes
- **Validation-only**: Fail fast with strict rule enforcement
- **Critic Mode**: Attempt to improve content that fails validation

### Rule Categories
- **Content Rules**: Length, prohibited terms, formatting
- **Pattern Rules**: Symmetry, repetition, structure
- **Safety Rules**: Toxicity, bias, harmful content
- **Domain Rules**: Legal, medical, code validation

### Provider Support
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- LLaMA, Mistral, etc.
- Custom providers

## Installation

```bash
pip install sifaka
```

For optional dependencies:
```bash
pip install sifaka[openai]     # OpenAI support
pip install sifaka[anthropic]  # Anthropic support
pip install sifaka[all]        # All providers
```

## Usage Examples

### Basic Validation

```python
from sifaka.chain import Chain
from sifaka.models import OpenAIProvider
from sifaka.rules import LengthRule

chain = Chain(
    model=OpenAIProvider(model_name="gpt-4", config={"api_key": "YOUR_API_KEY"}),
    rules=[LengthRule(min_length=50, max_length=200)],
    max_attempts=1  # Validation only
)

result = chain.run("Write a short summary of quantum mechanics.")
```

### Content Improvement

```python
from sifaka.critics.prompt import PromptCritic, PromptCriticConfig

critic = PromptCritic(
    model=provider,
    config=PromptCriticConfig(
        name="editor",
        system_prompt="You are an expert editor specialized in scientific writing."
    )
)

chain = Chain(
    model=provider,
    rules=[length_rule, style_rule],
    critic=critic,
    max_attempts=3
)

result = chain.run("Explain the theory of relativity.")
```

## Migration Guide

### From 0.x to 1.0

The main change is the deprecation of the `Reflector` class:

```python
# OLD (0.x): Using Reflector
from sifaka.reflector import Reflector

reflector = Reflector(
    reflection_config=ReflectionConfig(
        mirror_mode="both",
        symmetry_threshold=0.8
    )
)

# NEW (1.0+): Using specialized pattern rules
from sifaka.rules import SymmetryRule, RepetitionRule
from sifaka.chain import Chain

symmetry_rule = SymmetryRule(
    name="symmetry_check",
    config=RuleConfig(
        metadata={
            "mirror_mode": "both",
            "symmetry_threshold": 0.8
        }
    )
)

chain = Chain(model=provider, rules=[symmetry_rule])
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
