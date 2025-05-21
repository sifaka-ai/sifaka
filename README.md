# Sifaka

Sifaka is an AI validation, improvement, and evaluation framework with built-in guardrails.

## Overview

Sifaka provides a framework for:
1. Validating AI-generated content against a set of criteria
2. Improving content that fails validation using critics
3. Persisting the state of the generation process for analysis and debugging

## Key Components

### Thought Container

The central state container (`Thought`) that passes information between components:
- Prompt and generated text
- Retrieved context
- Validation results
- Critic feedback

### Models

Models generate text based on prompts and can use the Thought container to access context.

### Validators

Validators check if generated text meets specific criteria.

### Critics

Critics provide feedback on how to improve text that fails validation.

### Retrievers

Retrievers fetch relevant information from external sources and are available to both models and critics.

### Persistence

Mechanisms for storing and retrieving Thoughts:
- JSON
- Redis
- Milvus
- Elasticsearch

## Installation

```bash
# Basic installation
pip install sifaka

# With all dependencies
pip install sifaka[all]

# With specific components
pip install sifaka[persistence]  # All persistence mechanisms
pip install sifaka[redis]        # Just Redis persistence
pip install sifaka[retrievers]   # All retriever mechanisms
```

## Usage

```python
from sifaka import Chain, Thought
from sifaka.models import OpenAIModel
from sifaka.validators import ProfanityValidator
from sifaka.critics import ReflexionCritic
from sifaka.persistence import JSONPersistence

# Create a chain with validators and critics
chain = Chain(
    model=OpenAIModel(),
    validators=[ProfanityValidator()],
    critics=[ReflexionCritic()],
    persistence=JSONPersistence(path="thoughts.json")
)

# Generate text
result = chain.generate("Write a short story about a robot.")

# Access the thought container
thought = result.thought
print(f"Generated text: {thought.text}")
print(f"Validation results: {thought.validation_results}")
print(f"Critic feedback: {thought.critic_feedback}")
```

## License

MIT
