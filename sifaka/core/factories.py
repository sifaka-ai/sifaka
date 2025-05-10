"""
Unified Factory Module for Sifaka

This module provides a unified interface for creating components in the Sifaka framework.
It centralizes factory functions from all components, providing a consistent way to
create and configure components.

## Usage Examples

```python
from sifaka.core.factories import (
    create_chain,
    create_critic,
    create_rule,
    create_classifier,
    create_retriever,
    create_adapter,
    create_model_provider
)

# Create a chain
chain = create_chain(
    chain_type="simple",
    model="gpt-4",
    rules=["length", "toxicity"],
    critic="prompt"
)

# Create a critic
critic = create_critic(
    critic_type="prompt",
    model="gpt-4",
    system_prompt="You are an expert editor."
)

# Create a rule
rule = create_rule(
    rule_type="length",
    min_length=10,
    max_length=1000
)

# Create a classifier
classifier = create_classifier(
    classifier_type="toxicity",
    threshold=0.7
)

# Create a retriever
retriever = create_retriever(
    retriever_type="simple",
    documents={"doc1": "content1", "doc2": "content2"},
    max_results=3
)

# Create an adapter
adapter = create_adapter(
    adapter_type="pydantic",
    rules=[rule],
    output_model=MyModel
)

# Create a model provider
model = create_model_provider(
    provider_type="openai",
    model_name="gpt-4",
    api_key="your-api-key"
)
```
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

# Import component-specific factory functions
from sifaka.chain.factories import create_simple_chain, create_backoff_chain
from sifaka.critics.implementations import (
    create_prompt_critic,
    create_reflexion_critic,
    create_constitutional_critic,
    create_self_refine_critic,
    create_self_rag_critic,
    create_lac_critic,
)
from sifaka.rules.factories import create_rule as create_rule_base
from sifaka.classifiers.factories import (
    create_toxicity_classifier,
    create_sentiment_classifier,
    create_bias_detector,
    create_language_classifier,
    create_readability_classifier,
)
from sifaka.retrieval.factories import create_simple_retriever, create_threshold_retriever
from sifaka.adapters.base import create_adapter as create_adapter_base
from sifaka.models.factories import (
    create_model_provider as create_model_provider_base,
    create_openai_provider,
    create_anthropic_provider,
    create_gemini_provider,
)

# Type variables for return types
T = TypeVar("T")
ChainType = TypeVar("ChainType")
CriticType = TypeVar("CriticType")
RuleType = TypeVar("RuleType")
ClassifierType = TypeVar("ClassifierType")
RetrieverType = TypeVar("RetrieverType")
AdapterType = TypeVar("AdapterType")
ModelProviderType = TypeVar("ModelProviderType")


def create_chain(
    chain_type: str = "simple",
    model: Any = None,
    rules: Optional[List[Any]] = None,
    critic: Optional[Any] = None,
    max_attempts: int = 3,
    **kwargs: Any,
) -> Any:
    """
    Create a chain with the specified configuration.

    This function creates a chain of the specified type with the given configuration,
    simplifying the creation of chains with common configurations.

    Args:
        chain_type: The type of chain to create ("simple" or "backoff")
        model: The model provider to use
        rules: The rules to validate against
        critic: The critic to use for refinement
        max_attempts: Maximum number of attempts
        **kwargs: Additional keyword arguments for the chain

    Returns:
        A chain instance

    Raises:
        ValueError: If the chain type is invalid or required parameters are missing
    """
    # Validate chain type
    if chain_type not in ["simple", "backoff"]:
        raise ValueError(f"Invalid chain type: {chain_type}")

    # Create chain based on type
    if chain_type == "simple":
        return create_simple_chain(
            model=model,
            rules=rules or [],
            critic=critic,
            max_attempts=max_attempts,
            **kwargs,
        )
    else:  # chain_type == "backoff"
        return create_backoff_chain(
            model=model,
            rules=rules or [],
            critic=critic,
            max_attempts=max_attempts,
            **kwargs,
        )


def create_critic(
    critic_type: str,
    llm_provider: Any = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a critic with the specified configuration.

    This function creates a critic of the specified type with the given configuration,
    simplifying the creation of critics with common configurations.

    Args:
        critic_type: The type of critic to create
        llm_provider: The language model provider to use
        name: Optional name for the critic
        description: Optional description for the critic
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A critic instance

    Raises:
        ValueError: If the critic type is invalid or required parameters are missing
    """
    # Set default name and description based on critic type
    name = name or f"{critic_type}_critic"
    description = description or f"{critic_type.capitalize()} critic"

    # Create critic based on type
    if critic_type == "prompt":
        return create_prompt_critic(
            llm_provider=llm_provider,
            name=name,
            description=description,
            **kwargs,
        )
    elif critic_type == "reflexion":
        return create_reflexion_critic(
            llm_provider=llm_provider,
            name=name,
            description=description,
            **kwargs,
        )
    elif critic_type == "constitutional":
        return create_constitutional_critic(
            llm_provider=llm_provider,
            name=name,
            description=description,
            **kwargs,
        )
    elif critic_type == "self_refine":
        return create_self_refine_critic(
            llm_provider=llm_provider,
            name=name,
            description=description,
            **kwargs,
        )
    elif critic_type == "self_rag":
        return create_self_rag_critic(
            llm_provider=llm_provider,
            name=name,
            description=description,
            **kwargs,
        )
    elif critic_type == "lac":
        return create_lac_critic(
            llm_provider=llm_provider,
            name=name,
            description=description,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid critic type: {critic_type}")


def create_rule(
    rule_type: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a rule with the specified configuration.

    This function creates a rule of the specified type with the given configuration,
    simplifying the creation of rules with common configurations.

    Args:
        rule_type: The type of rule to create
        name: Optional name for the rule
        description: Optional description for the rule
        **kwargs: Additional keyword arguments for the rule

    Returns:
        A rule instance

    Raises:
        ValueError: If the rule type is invalid or required parameters are missing
    """
    # Set default name and description based on rule type
    name = name or f"{rule_type}_rule"
    description = description or f"{rule_type.capitalize()} rule"

    # Create rule using the base factory function
    return create_rule_base(
        rule_type=rule_type,
        name=name,
        description=description,
        **kwargs,
    )


def create_classifier(
    classifier_type: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a classifier with the specified configuration.

    This function creates a classifier of the specified type with the given configuration,
    simplifying the creation of classifiers with common configurations.

    Args:
        classifier_type: The type of classifier to create
        name: Optional name for the classifier
        description: Optional description for the classifier
        **kwargs: Additional keyword arguments for the classifier

    Returns:
        A classifier instance

    Raises:
        ValueError: If the classifier type is invalid or required parameters are missing
    """
    # Set default name and description based on classifier type
    name = name or f"{classifier_type}_classifier"
    description = description or f"{classifier_type.capitalize()} classifier"

    # Create classifier based on type
    if classifier_type == "toxicity":
        return create_toxicity_classifier(
            name=name,
            description=description,
            **kwargs,
        )
    elif classifier_type == "sentiment":
        return create_sentiment_classifier(
            name=name,
            description=description,
            **kwargs,
        )
    elif classifier_type == "bias":
        return create_bias_detector(
            name=name,
            description=description,
            **kwargs,
        )
    elif classifier_type == "language":
        return create_language_classifier(
            name=name,
            description=description,
            **kwargs,
        )
    elif classifier_type == "readability":
        return create_readability_classifier(
            name=name,
            description=description,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid classifier type: {classifier_type}")


def create_retriever(
    retriever_type: str = "simple",
    documents: Optional[Dict[str, str]] = None,
    corpus: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a retriever with the specified configuration.

    This function creates a retriever of the specified type with the given configuration,
    simplifying the creation of retrievers with common configurations.

    Args:
        retriever_type: The type of retriever to create
        documents: Optional dictionary of documents
        corpus: Optional corpus file path
        name: Optional name for the retriever
        description: Optional description for the retriever
        **kwargs: Additional keyword arguments for the retriever

    Returns:
        A retriever instance

    Raises:
        ValueError: If the retriever type is invalid or required parameters are missing
    """
    # Set default name and description based on retriever type
    name = name or f"{retriever_type}_retriever"
    description = description or f"{retriever_type.capitalize()} retriever"

    # Create retriever based on type
    if retriever_type == "simple":
        return create_simple_retriever(
            documents=documents,
            corpus=corpus,
            name=name,
            description=description,
            **kwargs,
        )
    elif retriever_type == "threshold":
        return create_threshold_retriever(
            documents=documents,
            corpus=corpus,
            name=name,
            description=description,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid retriever type: {retriever_type}")


def create_adapter(
    adapter_type: str,
    adaptee: Any = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create an adapter with the specified configuration.

    This function creates an adapter of the specified type with the given configuration,
    simplifying the creation of adapters with common configurations.

    Args:
        adapter_type: The type of adapter to create
        adaptee: The component to adapt
        name: Optional name for the adapter
        description: Optional description for the adapter
        **kwargs: Additional keyword arguments for the adapter

    Returns:
        An adapter instance

    Raises:
        ValueError: If the adapter type is invalid or required parameters are missing
    """
    # Set default name and description based on adapter type
    name = name or f"{adapter_type}_adapter"
    description = description or f"{adapter_type.capitalize()} adapter"

    # Create adapter using the base factory function
    return create_adapter_base(
        adapter_type=adapter_type,
        adaptee=adaptee,
        name=name,
        description=description,
        **kwargs,
    )


def create_model_provider(
    provider_type: str,
    model_name: str,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a model provider with the specified configuration.

    This function creates a model provider of the specified type with the given configuration,
    simplifying the creation of model providers with common configurations.

    Args:
        provider_type: The type of model provider to create
        model_name: The name of the model to use
        api_key: Optional API key for the model provider
        **kwargs: Additional keyword arguments for the model provider

    Returns:
        A model provider instance

    Raises:
        ValueError: If the provider type is invalid or required parameters are missing
    """
    # Create model provider based on type
    if provider_type == "openai":
        return create_openai_provider(
            model_name=model_name,
            api_key=api_key,
            **kwargs,
        )
    elif provider_type == "anthropic":
        return create_anthropic_provider(
            model_name=model_name,
            api_key=api_key,
            **kwargs,
        )
    elif provider_type == "gemini":
        return create_gemini_provider(
            model_name=model_name,
            api_key=api_key,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid provider type: {provider_type}")
