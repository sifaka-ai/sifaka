"""
Interfaces for Sifaka.

This package provides interfaces for components in the Sifaka framework.
These interfaces establish a common contract for component behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Chain**: Interface for chains
2. **Retriever**: Interface for retrievers
3. **Classifier**: Interface for classifiers
4. **Critic**: Interface for critics
5. **Rule**: Interface for rules
6. **Adapter**: Interface for adapters
7. **Model**: Interface for models

## Usage Examples

```python
from sifaka.interfaces import Chain, Retriever, Critic

# Create a chain implementation
class MyChain(Chain[str, str]):
    def run(self, input_value: str) -> str:
        return f"Processed: {input_value}"

# Create a retriever implementation
class MyRetriever(Retriever[str, list]):
    def retrieve(self, query: str) -> list:
        return [f"Result for {query}"]

# Create a critic implementation
class MyCritic(Critic[str, str]):
    def evaluate(self, input_value: str, output_value: str) -> bool:
        return len(output_value) > 0
```

## Error Handling

- ValueError: Raised for invalid inputs
- RuntimeError: Raised for execution failures
- TypeError: Raised for type mismatches
"""

# Import interfaces from component-specific directories
from .chain import Chain, AsyncChain
from .retrieval import Retriever, AsyncRetriever

__all__ = [
    "Chain",
    "AsyncChain",
    "Retriever",
    "AsyncRetriever",
]
