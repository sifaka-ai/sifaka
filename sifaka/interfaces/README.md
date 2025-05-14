# Sifaka Interfaces

This package provides the interfaces that define the contracts for components in the Sifaka framework. These interfaces establish a common set of behaviors and capabilities that components must implement, enabling better modularity, extensibility, and interoperability.

## Architecture

The interfaces architecture follows a domain-driven design with both synchronous and asynchronous variants:

```
Interfaces
├── Core
│   ├── Chain (orchestration of components)
│   ├── Model (text generation)
│   ├── Critic (text validation and improvement)
│   ├── Rule (validation rules)
│   ├── Classifier (text classification)
│   └── Retriever (information retrieval)
├── Protocol Types
│   ├── Synchronous Interfaces (standard operation)
│   └── Asynchronous Interfaces (non-blocking operation)
└── Specialized
    ├── Adapter (component adaptation)
    ├── Client (API interactions)
    └── Counter (token counting)
```

## Key Interfaces

### Chain Interfaces

Interfaces for components that orchestrate the flow between models, validators, and improvers:

- **Chain**: Main interface for running a sequence of operations
- **AsyncChain**: Asynchronous variant of Chain
- **Components**: Model, Validator, Improver, Formatter

### Critic Interfaces

Interfaces for components that validate, critique, and improve text:

- **Critic**: Interface for text validation and improvement
- **AsyncCritic**: Asynchronous variant of Critic
- **TextValidator**: Interface for validating text quality
- **TextImprover**: Interface for improving text based on feedback
- **TextCritic**: Interface for critiquing text and providing feedback

### Model Interfaces

Interfaces for text generation models:

- **ModelProviderProtocol**: Interface for model providers
- **AsyncModelProviderProtocol**: Asynchronous variant of ModelProviderProtocol
- **LLMProvider**: Interface for language model providers

### Rule Interfaces

Interfaces for validation rules:

- **Rule**: Interface for components that validate text against specific criteria
- **AsyncRule**: Asynchronous variant of Rule
- **RuleProtocol**: Lower-level protocol for rule implementation
- **Validatable**: Interface for objects that can be validated

### Classifier Interfaces

Interfaces for text classification:

- **ClassifierProtocol**: Interface for text classifiers
- **TextProcessor**: Interface for text processing operations

### Retriever Interfaces

Interfaces for information retrieval:

- **Retriever**: Interface for retrieving information
- **AsyncRetriever**: Asynchronous variant of Retriever
- **DocumentStore**: Interface for storing and retrieving documents
- **IndexManager**: Interface for managing search indices
- **QueryProcessor**: Interface for processing search queries

## Usage

### Implementing Interfaces

To implement an interface, create a class that inherits from the interface and implements all required methods:

```python
from sifaka.interfaces import Critic, CritiqueResult

class MyCritic(Critic):
    def __init__(self, name: str, config: dict = None):
        self._name = name
        self._config = config or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A custom critic implementation"

    @property
    def config(self) -> dict:
        return self._config

    def update_config(self, config: dict) -> None:
        self._config.update(config)

    def validate(self, text: str) -> bool:
        # Implement validation logic
        return len(text) > 10

    def critique(self, text: str) -> CritiqueResult:
        # Implement critique logic
        return {
            "score": 0.8 if len(text) > 20 else 0.5,
            "feedback": "Text length is acceptable" if len(text) > 20 else "Text is too short",
            "issues": [] if len(text) > 20 else ["Text is too short"],
            "suggestions": [] if len(text) > 20 else ["Add more content"]
        }

    def improve(self, text: str, feedback: str = None) -> str:
        # Implement improvement logic
        if not feedback:
            return text
        if "short" in feedback and len(text) < 20:
            return text + " Additional content to make the text longer."
        return text
```

### Using Protocol Types

Protocol types can be used for type checking and static analysis:

```python
from typing import List
from sifaka.interfaces import TextValidator, TextImprover, CritiqueResult

def process_text(validator: TextValidator, improver: TextImprover, text: str) -> str:
    """Process text using validator and improver."""
    if not validator.validate(text):
        # Text needs improvement
        return improver.improve(text, "Improve text quality")
    return text

def analyze_results(results: List[CritiqueResult]) -> float:
    """Calculate average score from critique results."""
    if not results:
        return 0.0
    return sum(result["score"] for result in results) / len(results)
```

### Using Asynchronous Interfaces

Asynchronous interfaces can be used for non-blocking operations:

```python
import asyncio
from sifaka.interfaces import AsyncCritic

async def process_texts(critic: AsyncCritic, texts: list[str]) -> list[str]:
    """Process multiple texts asynchronously."""
    results = []
    for text in texts:
        if not await critic.validate(text):
            # Text needs improvement
            improved = await critic.improve(text, "Improve quality")
            results.append(improved)
        else:
            results.append(text)
    return results

# Run asynchronous function
texts = ["Text 1", "Text 2", "Text 3"]
asyncio.run(process_texts(my_async_critic, texts))
```

## Interface Design Principles

Sifaka interfaces follow these design principles:

1. **Separation of Concerns**: Each interface focuses on a specific responsibility
2. **Interface Segregation**: Smaller, focused interfaces over large, monolithic ones
3. **Protocol Support**: Uses Python's Protocol class for structural typing
4. **Async/Sync Variants**: Both synchronous and asynchronous variants where appropriate
5. **Type Hinting**: Strong type hints for better developer experience
6. **Method Documentation**: Detailed docstrings for all methods
7. **Error Handling**: Standardized approach to error handling
8. **Extensibility**: Designed for extension without modification
9. **Composition**: Supports building complex systems through composition

## Extension Patterns

To extend the interface system, follow these patterns:

### Adding New Methods

For additional functionality, create a new interface that extends the base interface:

```python
from sifaka.interfaces import Critic, Protocol, runtime_checkable

@runtime_checkable
class EnhancedCritic(Critic, Protocol):
    """Enhanced critic with additional methods."""

    def explain(self, text: str) -> str:
        """Explain why the text is valid or invalid."""
        ...

    def summarize(self, text: str) -> str:
        """Generate a summary of the text."""
        ...
```

### Creating Specialized Interfaces

For specialized domains, create domain-specific interfaces:

```python
from sifaka.interfaces import TextValidator, Protocol, runtime_checkable

@runtime_checkable
class CodeValidator(TextValidator, Protocol):
    """Validator specifically for code."""

    def validate_syntax(self, code: str, language: str) -> bool:
        """Validate code syntax."""
        ...

    def validate_style(self, code: str, style_guide: str) -> bool:
        """Validate code against style guide."""
        ...
```