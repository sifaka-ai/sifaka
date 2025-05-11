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
from .chain import (
    Chain,
    AsyncChain,
    ChainComponent,
    ValidationResult,
    Model,
    Validator,
    Improver,
    ChainFormatter,
    ChainPlugin,
    PromptManager,
    ValidationManager,
    RetryStrategy,
    ResultFormatter,
)
from .retrieval import Retriever, AsyncRetriever, DocumentStore, IndexManager, QueryProcessor
from .classifier import ClassifierProtocol, TextProcessor
from .critic import (
    Critic,
    AsyncCritic,
    CritiqueResult,
    TextValidator,
    AsyncTextValidator,
    SyncTextValidator,
    TextImprover,
    AsyncTextImprover,
    SyncTextImprover,
    TextCritic,
    AsyncTextCritic,
    SyncTextCritic,
    LLMProvider,
    AsyncLLMProvider,
    SyncLLMProvider,
    PromptFactory,
    AsyncPromptFactory,
    SyncPromptFactory,
)
from .adapter import Adaptable
from .model import ModelProviderProtocol, AsyncModelProviderProtocol
from .client import APIClientProtocol
from .counter import TokenCounterProtocol
from .rule import Rule, AsyncRule, RuleProtocol, RuleResultHandler, Validatable

__all__ = [
    # Chain interfaces
    "Chain",
    "AsyncChain",
    "ChainComponent",
    "ValidationResult",
    "Model",
    "Validator",
    "Improver",
    "ChainFormatter",
    "ChainPlugin",
    "PromptManager",
    "ValidationManager",
    "RetryStrategy",
    "ResultFormatter",
    # Retrieval interfaces
    "Retriever",
    "AsyncRetriever",
    "DocumentStore",
    "IndexManager",
    "QueryProcessor",
    # Classifier interfaces
    "ClassifierProtocol",
    "TextProcessor",
    # Critic interfaces
    "Critic",
    "AsyncCritic",
    "CritiqueResult",
    "TextValidator",
    "AsyncTextValidator",
    "SyncTextValidator",
    "TextImprover",
    "AsyncTextImprover",
    "SyncTextImprover",
    "TextCritic",
    "AsyncTextCritic",
    "SyncTextCritic",
    "LLMProvider",
    "AsyncLLMProvider",
    "SyncLLMProvider",
    "PromptFactory",
    "AsyncPromptFactory",
    "SyncPromptFactory",
    # Adapter interfaces
    "Adaptable",
    # Model interfaces
    "ModelProviderProtocol",
    "AsyncModelProviderProtocol",
    "APIClientProtocol",
    "TokenCounterProtocol",
    # Rule interfaces
    "Rule",
    "AsyncRule",
    "RuleProtocol",
    "RuleResultHandler",
    "Validatable",
]
