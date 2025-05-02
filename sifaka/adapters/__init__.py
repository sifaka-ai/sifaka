"""
Adapters for external libraries and services integration with Sifaka.

This package provides adapter implementations that allow Sifaka to integrate with
various external libraries, services, and model providers.

## Architecture Overview

The adapters module follows an adapter pattern to enable Sifaka to work with
external libraries and services without tight coupling:

1. **Protocol Definitions**: Define the expected interface for integration
2. **Adapter Components**: Implement wrappers that translate between systems
3. **Factory Functions**: Provide simple creation patterns for common use cases

## Component Types

Sifaka provides several types of adapters:

1. **LangChain Adapters**: Integration with LangChain chains, agents, and memory
2. **LangGraph Adapters**: Integration with LangGraph state graphs and nodes
3. **Rules Adapters**: Adapters that convert external components to Sifaka rules
4. **Classifier Adapters**: Special adapters for text classification systems

## Usage Patterns

The recommended way to use adapters is through factory functions:

```python
# LangChain example
from langchain.chains import LLMChain
from sifaka.adapters import wrap_chain, create_simple_langchain
from sifaka.rules.content import create_sentiment_rule

# Create rules
rule = create_sentiment_rule(valid_labels=["positive", "neutral"])

# Method 1: Create a wrapped chain from existing components
chain = LLMChain(...)
wrapped_chain = wrap_chain(chain, rules=[rule])

# Method 2: Create a chain with integrated Sifaka features
sifaka_chain = create_simple_langchain(
    llm=llm,
    prompt=prompt,
    rules=[rule],
    critique=True
)

# LangGraph example
from langgraph.graph import StateGraph
from sifaka.adapters import wrap_graph

graph = StateGraph()
# ... define graph nodes and edges

sifaka_graph = wrap_graph(
    graph=graph,
    rules=[rule]
)
```

For details on specific adapter implementations, see the respective module documentation.
"""

# LangChain adapters
from sifaka.adapters.langchain import (
    ChainValidator,
    ChainOutputProcessor,
    ChainMemory,
    ChainConfig,
    SifakaChain,
    RuleBasedValidator as LangChainRuleBasedValidator,
    SifakaMemory,
    create_simple_langchain,
    wrap_chain,
    wrap_memory,
)

# LangGraph adapters
from sifaka.adapters.langgraph import (
    GraphValidator,
    GraphProcessor,
    GraphNode,
    GraphConfig,
    SifakaGraph,
    SifakaNode,
    RuleBasedValidator as LangGraphRuleBasedValidator,
    create_simple_graph,
    wrap_graph,
    wrap_node,
)

# Rules adapters
from sifaka.adapters.rules import (
    Adaptable,
    BaseAdapter,
    ClassifierAdapter,
    ClassifierRule,
    create_classifier_rule,
)

# Export types from sifaka.classifiers.base for convenience
from sifaka.classifiers.base import ClassificationResult, ClassifierConfig

# Try to import Guardrails adapters if available
try:
    from sifaka.adapters.rules import (
        GuardrailsValidatorAdapter,
        create_guardrails_rule,
    )
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False

__all__ = [
    # LangChain adapters
    "ChainValidator",
    "ChainOutputProcessor",
    "ChainMemory",
    "ChainConfig",
    "SifakaChain",
    "LangChainRuleBasedValidator",
    "SifakaMemory",
    "create_simple_langchain",
    "wrap_chain",
    "wrap_memory",

    # LangGraph adapters
    "GraphValidator",
    "GraphProcessor",
    "GraphNode",
    "GraphConfig",
    "SifakaGraph",
    "SifakaNode",
    "LangGraphRuleBasedValidator",
    "create_simple_graph",
    "wrap_graph",
    "wrap_node",

    # Rules adapters
    "Adaptable",
    "BaseAdapter",
    "ClassifierAdapter",
    "ClassifierRule",
    "create_classifier_rule",

    # Classifier types
    "ClassificationResult",
    "ClassifierConfig",
]

# Add Guardrails adapter to exports if available
if GUARDRAILS_AVAILABLE:
    __all__.extend(
        [
            "GuardrailsValidatorAdapter",
            "create_guardrails_rule",
        ]
    )