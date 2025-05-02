"""
Adapters for external libraries and services integration with Sifaka.

This package provides adapter implementations that allow Sifaka to integrate with
various external libraries, services, and model providers.
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