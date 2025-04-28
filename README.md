# Sifaka: Reflection and Reliability for LLMs

**Sifaka** is an open-source framework that adds **reflection and reliability** to LLM applications. Build safer, more reliable AI systems that:

- ✅ Catch hallucinations before they reach users
- ✅ Enforce rules and tone consistency
- ✅ Provide transparency and auditability
- ✅ Analyze text characteristics and patterns

[![PyPI version](https://badge.fury.io/py/sifaka.svg)](https://badge.fury.io/py/sifaka)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

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
    LengthRule(
        name="length",
        config={
            "min_length": 50,
            "max_length": 500,
            "unit": "characters",
            "priority": 2,
            "cost": 1.5,
        }
    ),
    ProhibitedContentRule(
        name="content_filter",
        config={
            "prohibited_terms": ["inappropriate", "offensive"],
            "case_sensitive": False,
            "priority": 2,
            "cost": 1.5,
        }
    )
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

Sifaka has four main components:

### 1. Chain
The orchestrator that manages the validation and improvement cycle:

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
Enforce constraints on content with specialized validators:

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

# Pattern Rules Example
from sifaka.rules.base import RuleConfig, RulePriority

symmetry_rule = SymmetryRule(
    name="symmetry_check",
    description="Checks for text symmetry patterns",
    config=RuleConfig(
        priority=RulePriority.MEDIUM,
        metadata={
            "symmetry_threshold": 0.8,
            "preserve_whitespace": True,
            "preserve_case": True,
            "ignore_punctuation": True,
        }
    )
)

repetition_rule = RepetitionRule(
    name="repetition_check",
    description="Detects repetitive patterns",
    config=RuleConfig(
        priority=RulePriority.MEDIUM,
        metadata={
            "pattern_type": "repeat",
            "pattern_length": 3,
            "case_sensitive": True,
            "allow_overlap": False,
        }
    )
)
```

### 3. Critics
Analyze and improve content based on rule failures:

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
Analyze text characteristics for various properties:

```python
from sifaka.classifiers import (
    SentimentClassifier,     # Analyze sentiment
    ReadabilityClassifier,   # Assess reading level
    ToxicityClassifier,      # Detect toxic content
    BiasDetector,            # Detect bias in text
    GenreClassifier          # Classify text genre
)

# Direct usage
sentiment = SentimentClassifier()
result = sentiment.classify("This is amazing!")
print(result.label)         # "positive"
print(result.confidence)    # confidence score

# With readability metrics
readability = ReadabilityClassifier()
result = readability.classify("Simple text to analyze.")
print(result.label)         # "elementary"
print(result.metadata)      # Contains Flesch-Kincaid scores

# Or use with a rule
from sifaka.rules import ClassifierRule
sentiment_rule = ClassifierRule(classifier=SentimentClassifier())
```

## Key Features

### 1. Text Analysis
- Sentiment analysis with confidence scores
- Readability assessment with Flesch-Kincaid metrics
- Genre classification
- Bias detection
- Emotional content analysis

### 2. Pattern Detection
- Symmetry analysis
- Repetition detection
- Structure validation
- Custom pattern matching

### 3. Content Validation
- Length constraints
- Prohibited content filtering
- Format validation
- Toxicity detection
- Bias checking

### 4. Content Improvement
- Automated content refinement
- Rule-based improvements
- Tone consistency enforcement
- Reading level adjustment

### 5. Validation Modes
- **Validation-only**: Fail fast with strict rule enforcement
- **Critic Mode**: Attempt to improve content that fails validation

## Installation

```bash
# Requirements
Python 3.11 or higher

# Basic installation
pip install sifaka

# With toxicity detection
pip install sifaka[toxicity]

# With all extras
pip install sifaka[all]
```

## Usage Examples

The `examples/` directory contains several example scripts:

1. `usage.py`: Basic usage demonstrating:
   - Sentiment analysis
   - Emotional content analysis
   - Readability analysis
   - Text improvement

2. `advanced_classifiers_example.py`: Advanced usage showing:
   - Genre classification
   - Bias detection
   - Rule adapter functionality

3. `combined_classifiers.py`: Integration example with:
   - Multiple classifier combination
   - Pattern detection
   - Rule validation
   - Metrics analysis

⚠️ **Note**: The `Reflector` class is deprecated and will be removed in version 2.0.0. Use `SymmetryRule` and `RepetitionRule` from `sifaka.rules.pattern_rules` instead.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
