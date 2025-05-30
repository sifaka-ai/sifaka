# Sifaka Documentation

Sifaka is a PydanticAI-native AI validation framework for building reliable text generation workflows.

## Quick Start

```python
from pydantic_ai import Agent
from sifaka.agents import create_pydantic_chain
from sifaka.validators import LengthValidator
from sifaka.critics import ReflexionCritic
from sifaka.models import create_model

# Create PydanticAI agent
agent = Agent("openai:gpt-4", system_prompt="You are a helpful assistant.")

# Create Sifaka components
validator = LengthValidator(min_length=50, max_length=500)
critic = ReflexionCritic(model=create_model("openai:gpt-3.5-turbo"))

# Create chain
chain = create_pydantic_chain(
    agent=agent,
    validators=[validator],
    critics=[critic]
)

# Run the chain
result = chain.run("Write a story about AI.")
print(result.text)
```

## Installation

```bash
pip install sifaka[openai]  # For OpenAI models
pip install sifaka[anthropic]  # For Anthropic models
pip install sifaka[all]  # For all providers
```

## Core Components

### Validators
Validate generated text:
- `LengthValidator` - Check text length
- `RegexValidator` - Pattern matching
- `ContentValidator` - Keyword requirements

### Critics
Improve text quality:
- `ReflexionCritic` - General improvement feedback
- `SelfRefineCritic` - Iterative refinement
- `ConstitutionalCritic` - Principle-based feedback

### Storage
Store conversation history:
- `MemoryStorage` - In-memory (development)
- `FileStorage` - Local files (simple persistence)
- `RedisStorage` - Redis backend (production)

## Examples

### Basic Validation
```python
from sifaka.validators import LengthValidator

validator = LengthValidator(min_length=100, max_length=1000)
chain = create_pydantic_chain(agent=agent, validators=[validator])
```

### Multiple Critics
```python
from sifaka.critics import ReflexionCritic, SelfRefineCritic

critics = [
    ReflexionCritic(model=create_model("openai:gpt-3.5-turbo")),
    SelfRefineCritic(model=create_model("openai:gpt-3.5-turbo"))
]
chain = create_pydantic_chain(agent=agent, critics=critics)
```

### With Storage
```python
from sifaka.storage import FileStorage

storage = FileStorage("thoughts.json")
chain = create_pydantic_chain(
    agent=agent,
    validators=[validator],
    critics=[critic],
    storage=storage
)
```

## API Reference

See the source code for detailed API documentation. All classes and functions include comprehensive docstrings.
