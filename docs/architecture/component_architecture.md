# Component Architecture

This document describes the component architecture of the Sifaka framework.

## Core Components

Sifaka is built around a set of key components:

1. **Rules**: Enforce constraints on model outputs
2. **Critics**: Provide feedback to improve generated content
3. **Chain**: Orchestrates the model, validation, and improvement loop
4. **Models**: Provide access to language model APIs
5. **Classifiers**: Analyze text for specific characteristics

### How Components Interact

```
┌───────────┐    Generate    ┌─────────┐
│           │───────────────>│         │
│   Chain   │                │  Model  │
│           │<───────────────│         │
└───────────┘    Response    └─────────┘
      │
      │ Validate
      ▼
┌───────────┐    Pass/Fail   ┌─────────┐
│           │───────────────>│         │
│   Rules   │                │  Chain  │
│           │<───────────────│         │
└───────────┘    Feedback    └─────────┘
      │
      │ If fail, improve
      ▼
┌───────────┐    Feedback    ┌─────────┐
│           │───────────────>│         │
│  Critics  │                │  Chain  │
│           │<───────────────│         │
└───────────┘    Learning    └─────────┘
```

## Directory Structure

The Sifaka framework follows a modular structure:

```
sifaka/
├── chain/           # Chain orchestration logic
│   ├── formatters/  # Output formatters
│   ├── managers/    # Component managers
│   └── strategies/  # Retry strategies
├── classifiers/     # Text classification modules
├── critics/         # Feedback and improvement
├── models/          # LLM provider integrations
├── rules/           # Validation rule components
│   ├── content/     # Content-based rules
│   ├── domain/      # Domain-specific rules
│   ├── factual/     # Factual validation
│   └── formatting/  # Format-related rules
├── adapters/        # Framework integrations
└── utils/           # Common utilities
```

## Component Details

### Rules

Rules validate model outputs against specific criteria and are organized into categories:

- `rules/content/`: Rules related to content safety, sentiment, etc.
- `rules/domain/`: Domain-specific rules for legal, medical, etc.
- `rules/factual/`: Rules that verify factual accuracy
- `rules/formatting/`: Rules that enforce text formatting conventions

Key rule modules:
- `rules/base.py` - Base rule interfaces and abstract classes
- `rules/formatting/length.py` - Enforce content length constraints
- `rules/content/safety.py` - Content moderation and safety filters
- `rules/content/tone.py` - Tone and style enforcement

### Critics

Critics analyze model outputs and provide feedback for improvement:

- `critics/prompt.py` - Standard prompt-based critic
- `critics/reflexion.py` - Critic with reflection capabilities
- `critics/style.py` - Style-focused improvement critic

Key critic modules:
- `critics/base.py` - Base critic interfaces
- `critics/managers/` - Managers for critic components
- `critics/services/` - Services for critic functionality

### Chain

Chain components orchestrate the model, validation, and improvement process:

- `chain/core.py` - Core chain functionality
- `chain/orchestrator.py` - High-level chain orchestration
- `chain/strategies/` - Different chain execution strategies

### Models

Model components provide abstract interfaces to LLM providers:

- `models/anthropic.py` - Anthropic Claude integration
- `models/openai.py` - OpenAI integration
- `models/gemini.py` - Google Gemini integration

### Adapters

The adapters module provides integration with external frameworks:

- `adapters/__init__.py` - Adapter initialization
- `adapters/rules/` - Adapters for rule integration
- `adapters/rules/classifier.py` - Adapt classifiers as rules
- `adapters/rules/guardrails_adapter.py` - Integration with Guardrails

### Classifiers

Classifiers analyze text for specific characteristics:

- `classifiers/toxicity.py` - Detect toxic content
- `classifiers/sentiment.py` - Analyze sentiment
- `classifiers/language.py` - Detect language
- `classifiers/profanity.py` - Detect profanity

### Utilities

Utility modules provide common functionality:

- `utils/logging.py` - Logging utilities
- `utils/tracing.py` - Tracing and observability
- `utils/validation.py` - Input validation helpers

## Component Relationships

### Chain and Model Relationship

The Chain orchestrates the interaction with language models:

```python
from sifaka.chain import ChainOrchestrator
from sifaka.models.anthropic import AnthropicProvider

model = AnthropicProvider()
chain = ChainOrchestrator(model=model)
```

### Chain and Rules Relationship

Chains use Rules to validate model outputs:

```python
from sifaka.chain import ChainOrchestrator
from sifaka.rules.formatting.length import create_length_rule

rule = create_length_rule(min_words=100, max_words=500)
chain = ChainOrchestrator(model=model, rules=[rule])
```

### Chain and Critics Relationship

Chains use Critics to improve model outputs:

```python
from sifaka.chain import ChainOrchestrator
from sifaka.critics.prompt import PromptCritic

critic = PromptCritic(llm_provider=model)
chain = ChainOrchestrator(model=model, rules=[rule], critic=critic)
```

## Further Reading

- [System Design](system_design.md) - Higher-level system design
- [Rules and Validators](../rules_and_validators.md) - Detailed explanation of rules
- [Chain Architecture](../chain_architecture.md) - Chain components and patterns