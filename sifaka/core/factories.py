"""
Unified Factory Module for Sifaka

This module provides a unified interface for creating components in the Sifaka framework.
It centralizes factory functions from all components, providing a consistent way to
create and configure components.

## Overview
The factory module serves as a central point for creating all Sifaka components,
abstracting away the complexity of component initialization and dependency resolution.
It provides a consistent interface for creating chains, critics, rules, classifiers,
retrievers, adapters, and model providers.

## Components
- **Core Factory Functions**: High-level factory functions for creating components
- **Rule Factory Functions**: Specialized factory functions for creating rules
- **Critic Factory Functions**: Specialized factory functions for creating critics
- **Model Provider Factory Functions**: Specialized factory functions for creating model providers

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

## Error Handling
Factory functions validate inputs and raise appropriate exceptions when invalid
parameters are provided. Common exceptions include:
- `ValueError`: When required parameters are missing or invalid
- `DependencyError`: When dependencies cannot be resolved
- `ConfigurationError`: When component configuration is invalid

## Configuration
Factory functions accept configuration parameters as keyword arguments, which are
passed to the underlying component constructors. Common configuration parameters
include:
- `name`: Optional name for the component
- `description`: Optional description for the component
- `session_id`: Optional session ID for session-scoped dependencies
- `request_id`: Optional request ID for request-scoped dependencies
"""

from typing import Any, Dict, List, Optional, TypeVar

__all__ = [
    # Core factory functions
    "create_chain",
    "create_critic",
    "create_rule",
    "create_classifier",
    "create_retriever",
    "create_adapter",
    "create_model_provider",
    # Rule factory functions
    "create_length_rule",
    "create_prohibited_content_rule",
    "create_toxicity_rule",
    "create_bias_rule",
    "create_harmful_content_rule",
    "create_sentiment_rule",
    "create_structure_rule",
    "create_markdown_rule",
    "create_json_rule",
    "create_plain_text_rule",
    "create_format_rule",
    # Critic factory functions
    "create_prompt_critic",
    "create_reflexion_critic",
    "create_constitutional_critic",
    "create_self_refine_critic",
    "create_self_rag_critic",
    "create_lac_critic",
    # Model provider factory functions
    "create_openai_provider",
    "create_anthropic_provider",
    "create_gemini_provider",
]

# Import only what's needed at module level
from sifaka.chain.factories import create_simple_chain, create_backoff_chain
from sifaka.rules.factories import create_rule as create_rule_base
from sifaka.adapters.base import create_adapter as create_adapter_base

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
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a chain with the specified configuration.

    This function creates a chain of the specified type with the given configuration,
    simplifying the creation of chains with common configurations. It uses the
    dependency injection system to resolve dependencies if not explicitly provided.

    ## Architecture
    The factory function follows a standard pattern:
    1. Validate inputs
    2. Resolve dependencies using the dependency injection system
    3. Delegate to specific chain factory
    4. Return configured instance

    ## Lifecycle
    The created chain follows the standard component lifecycle:
    1. Initialization with provided configuration
    2. Warm-up (if needed)
    3. Operation
    4. Cleanup (when no longer needed)

    ## Error Handling
    - Raises `ValueError` if chain_type is invalid
    - Raises `DependencyError` if required dependencies cannot be resolved
    - Propagates exceptions from specific chain factories

    Args:
        chain_type: The type of chain to create ("simple" or "backoff")
        model: The model provider to use (injected if not provided)
        rules: The rules to validate against (injected if not provided)
        critic: The critic to use for refinement (injected if not provided)
        max_attempts: Maximum number of attempts
        session_id: Optional session ID for session-scoped dependencies
        request_id: Optional request ID for request-scoped dependencies
        **kwargs: Additional keyword arguments for the chain

    Returns:
        A chain instance

    Raises:
        ValueError: If the chain type is invalid or required parameters are missing
        DependencyError: If required dependencies cannot be resolved

    Example:
        ```python
        # Create a simple chain with default model and rules
        chain = create_chain()

        # Create a backoff chain with custom model and rules
        from sifaka.models.providers.openai import OpenAIProvider
        from sifaka.rules.implementations.length import LengthRule

        model = OpenAIProvider("gpt-4")
        rules = [LengthRule(min_length=10, max_length=1000)]

        chain = create_chain(
            chain_type="backoff",
            model=model,
            rules=rules,
            max_attempts=5
        )
        ```
    """
    from sifaka.core.dependency import DependencyProvider, DependencyError

    # Validate chain type
    if chain_type not in ["simple", "backoff"]:
        raise ValueError(f"Invalid chain type: {chain_type}")

    # Get dependency provider
    provider = DependencyProvider()

    # Resolve model if not provided
    if model is None:
        try:
            # Try to get by name first
            model = provider.get("model_provider", None, session_id, request_id)
        except DependencyError:
            try:
                # Try to get by type if not found by name
                from sifaka.interfaces.model import ModelProviderProtocol

                model = provider.get_by_type(ModelProviderProtocol, None, session_id, request_id)
            except (DependencyError, ImportError):
                # Don't raise here, let the specific factory function handle it
                pass

    # Resolve rules if not provided
    if rules is None:
        try:
            # Try to get by name
            rules = provider.get("rules", [], session_id, request_id)
        except DependencyError:
            # Use empty list as default
            rules = []

    # Resolve critic if not provided
    if critic is None:
        try:
            # Try to get by name
            critic = provider.get("critic", None, session_id, request_id)
        except DependencyError:
            # Critic is optional, so we can continue without it
            pass

    # Create chain based on type
    if chain_type == "simple":
        return create_simple_chain(
            model=model,
            rules=rules or [],
            critic=critic,
            max_attempts=max_attempts,
            session_id=session_id,
            request_id=request_id,
            **kwargs,
        )
    else:  # chain_type == "backoff"
        return create_backoff_chain(
            model=model,
            rules=rules or [],
            critic=critic,
            max_attempts=max_attempts,
            session_id=session_id,
            request_id=request_id,
            **kwargs,
        )


def create_critic(
    critic_type: str,
    model_provider: Any = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a critic with the specified configuration.

    This function creates a critic of the specified type with the given configuration,
    simplifying the creation of critics with common configurations. It uses the
    dependency injection system to resolve dependencies if not explicitly provided.

    Args:
        critic_type: The type of critic to create
        model_provider: The language model provider to use (injected if not provided)
        name: Optional name for the critic
        description: Optional description for the critic
        session_id: Optional session ID for session-scoped dependencies
        request_id: Optional request ID for request-scoped dependencies
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A critic instance

    Raises:
        ValueError: If the critic type is invalid or required parameters are missing
        DependencyError: If required dependencies cannot be resolved
    """
    from sifaka.core.dependency import DependencyProvider, DependencyError

    # Set default name and description based on critic type
    name = name or f"{critic_type}_critic"
    description = description or f"{critic_type.capitalize()} critic"

    # Get dependency provider
    provider = DependencyProvider()

    # Resolve model_provider if not provided
    if model_provider is None:
        try:
            # Try to get by name first
            model_provider = provider.get("model_provider", None, session_id, request_id)
        except DependencyError:
            try:
                # Try to get by type if not found by name
                from sifaka.interfaces.model import ModelProviderProtocol

                model_provider = provider.get_by_type(
                    ModelProviderProtocol, None, session_id, request_id
                )
            except (DependencyError, ImportError):
                # Don't raise here, let the specific factory function handle it
                pass

    # Import critic factory functions lazily to avoid circular dependencies
    from sifaka.critics.implementations import (
        create_prompt_critic,
        create_reflexion_critic,
        create_constitutional_critic,
        create_self_refine_critic,
        create_self_rag_critic,
        create_lac_critic,
    )

    # Create critic based on type
    if critic_type == "prompt":
        return create_prompt_critic(
            llm_provider=model_provider,
            name=name,
            description=description,
            session_id=session_id,
            request_id=request_id,
            **kwargs,
        )
    elif critic_type == "reflexion":
        return create_reflexion_critic(
            llm_provider=model_provider,
            name=name,
            description=description,
            session_id=session_id,
            request_id=request_id,
            **kwargs,
        )
    elif critic_type == "constitutional":
        return create_constitutional_critic(
            llm_provider=model_provider,
            name=name,
            description=description,
            session_id=session_id,
            request_id=request_id,
            **kwargs,
        )
    elif critic_type == "self_refine":
        return create_self_refine_critic(
            llm_provider=model_provider,
            name=name,
            description=description,
            session_id=session_id,
            request_id=request_id,
            **kwargs,
        )
    elif critic_type == "self_rag":
        return create_self_rag_critic(
            llm_provider=model_provider,
            name=name,
            description=description,
            session_id=session_id,
            request_id=request_id,
            **kwargs,
        )
    elif critic_type == "lac":
        return create_lac_critic(
            llm_provider=model_provider,
            name=name,
            description=description,
            session_id=session_id,
            request_id=request_id,
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
        rule_type: The type of rule to create (e.g., "length", "prohibited_content", "toxicity")
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

    # Import rule factory functions lazily to avoid circular dependencies
    if rule_type == "length":
        from sifaka.rules.formatting.length import create_length_rule

        return create_length_rule(
            name=name,
            description=description,
            **kwargs,
        )
    elif rule_type == "prohibited_content":
        from sifaka.rules.content.prohibited import create_prohibited_content_rule

        return create_prohibited_content_rule(
            name=name,
            description=description,
            **kwargs,
        )
    elif rule_type == "toxicity":
        from sifaka.rules.content.safety import create_toxicity_rule

        return create_toxicity_rule(
            name=name,
            description=description,
            **kwargs,
        )
    elif rule_type == "bias":
        from sifaka.rules.content.safety import create_bias_rule

        return create_bias_rule(
            name=name,
            description=description,
            **kwargs,
        )
    elif rule_type == "harmful_content":
        from sifaka.rules.content.safety import create_harmful_content_rule

        return create_harmful_content_rule(
            name=name,
            description=description,
            **kwargs,
        )
    elif rule_type == "sentiment":
        from sifaka.rules.content.sentiment import create_sentiment_rule

        return create_sentiment_rule(
            name=name,
            description=description,
            **kwargs,
        )
    elif rule_type == "structure":
        from sifaka.rules.formatting.structure import create_structure_rule

        return create_structure_rule(
            name=name,
            description=description,
            **kwargs,
        )
    elif rule_type == "markdown":
        from sifaka.rules.formatting.format import create_markdown_rule

        return create_markdown_rule(
            name=name,
            description=description,
            **kwargs,
        )
    elif rule_type == "json":
        from sifaka.rules.formatting.format import create_json_rule

        return create_json_rule(
            name=name,
            description=description,
            **kwargs,
        )
    elif rule_type == "plain_text":
        from sifaka.rules.formatting.format import create_plain_text_rule

        return create_plain_text_rule(
            name=name,
            description=description,
            **kwargs,
        )
    elif rule_type == "format":
        from sifaka.rules.formatting.format import create_format_rule

        return create_format_rule(
            name=name,
            description=description,
            **kwargs,
        )
    else:
        # For custom or unrecognized rule types, use the base factory function
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

    # Import classifier factory functions lazily to avoid circular dependencies
    from sifaka.classifiers.factories import (
        create_toxicity_classifier,
        create_sentiment_classifier,
        create_bias_detector,
        create_language_classifier,
        create_readability_classifier,
    )

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

    # Import retriever factory functions lazily to avoid circular dependencies
    from sifaka.retrieval.factories import create_simple_retriever, create_threshold_retriever

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
    initialize: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Create an adapter with the specified configuration.

    This function creates an adapter of the specified type with the given configuration,
    simplifying the creation of adapters with common configurations.

    ## Overview
    This function provides a unified interface for creating adapters of different types,
    delegating to the appropriate factory function based on the adapter type.

    ## Architecture
    The factory function follows a standard pattern:
    1. Validate inputs
    2. Set default name and description
    3. Delegate to specific adapter factory
    4. Return configured instance

    Args:
        adapter_type: The type of adapter to create
        adaptee: The component to adapt
        name: Optional name for the adapter
        description: Optional description for the adapter
        initialize: Whether to initialize the adapter immediately
        **kwargs: Additional keyword arguments for the adapter

    Returns:
        An adapter instance

    Raises:
        ValueError: If the adapter type is invalid or required parameters are missing

    ## Examples
    ```python
    # Create a classifier adapter
    from sifaka.classifiers.implementations.content.toxicity import create_toxicity_classifier

    classifier = create_toxicity_classifier()
    adapter = create_adapter(
        adapter_type="classifier",
        adaptee=classifier,
        threshold=0.8,
        valid_labels=["non_toxic"]
    )

    # Create a guardrails adapter
    from guardrails.hub import RegexMatch

    validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")
    adapter = create_adapter(
        adapter_type="guardrails",
        adaptee=validator,
        name="Phone Number Validator"
    )
    ```
    """
    # Validate inputs
    if not adapter_type:
        raise ValueError("adapter_type must be provided")

    # Set default name and description based on adapter type
    name = name or f"{adapter_type}_adapter"
    description = description or f"{adapter_type.capitalize()} adapter"

    # Create adapter using the base factory function
    return create_adapter_base(
        adapter_type=adapter_type,
        adaptee=adaptee,
        name=name,
        description=description,
        initialize=initialize,
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
    # Import model provider factory functions lazily to avoid circular dependencies
    from sifaka.models.factories import (
        create_openai_provider,
        create_anthropic_provider,
        create_gemini_provider,
    )

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
