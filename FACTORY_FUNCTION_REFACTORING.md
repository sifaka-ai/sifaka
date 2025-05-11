# Factory Function Refactoring Plan

This document outlines the plan for refactoring factory functions in the Sifaka codebase to improve dependency management.

## Current State

The Sifaka codebase currently has factory functions in various components:

1. `sifaka/core/factories.py`: Unified factory functions for all components
2. `sifaka/chain/factories.py`: Factory functions for chain components
3. `sifaka/critics/implementations/__init__.py`: Factory functions for critic components
4. `sifaka/models/factories.py`: Factory functions for model components
5. `sifaka/rules/factories.py`: Factory functions for rule components
6. `sifaka/retrieval/factories.py`: Factory functions for retrieval components

However, the current implementation has some limitations:

1. Inconsistent parameter naming across factory functions
2. No dependency resolution in factory functions
3. Limited validation for required dependencies
4. Inconsistent type annotations
5. Hard-coded dependencies in some factory functions

## Refactoring Goals

1. Standardize parameter naming across factory functions
2. Implement dependency resolution in factory functions
3. Add validation for required dependencies
4. Use type annotations consistently
5. Remove hard-coded dependencies

## Implementation Plan

### 1. Standardize Parameter Naming

#### 1.1 Define Standard Parameter Names

- `name`: The component name
- `description`: The component description
- `config`: The component configuration
- `model` or `model_provider`: The model provider
- `rules` or `validators`: The rules or validators
- `critic` or `improver`: The critic or improver
- `retriever`: The retriever
- `adapter`: The adapter
- `classifier`: The classifier
- `max_attempts`: Maximum number of attempts
- `**kwargs`: Additional keyword arguments

#### 1.2 Update Factory Functions

```python
# Before
def create_simple_chain(
    model: Any,
    rules: List[Any] = None,
    critic: Optional[Any] = None,
    max_attempts: int = 3,
    **kwargs: Any,
) -> Any:
    # ...

# After
def create_simple_chain(
    model_provider: Any,
    validators: List[Any] = None,
    improver: Optional[Any] = None,
    max_attempts: int = 3,
    name: str = "simple_chain",
    description: str = "Simple chain for text generation and validation",
    config: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    # ...
```

### 2. Implement Dependency Resolution

#### 2.1 Use DependencyProvider in Factory Functions

```python
def create_simple_chain(
    model_provider: Optional[Any] = None,
    validators: Optional[List[Any]] = None,
    improver: Optional[Any] = None,
    max_attempts: int = 3,
    name: str = "simple_chain",
    description: str = "Simple chain for text generation and validation",
    config: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a simple chain.

    Args:
        model_provider: The model provider to use
        validators: The validators to use
        improver: The improver to use
        max_attempts: Maximum number of attempts
        name: The chain name
        description: The chain description
        config: Optional chain configuration
        **kwargs: Additional keyword arguments

    Returns:
        A simple chain instance
    """
    # Get dependency provider
    provider = DependencyProvider()

    # Resolve dependencies
    if model_provider is None:
        try:
            model_provider = provider.get("model_provider")
        except DependencyError:
            raise ValueError("Model provider is required")

    if validators is None:
        try:
            validators = provider.get("validators", [])
        except DependencyError:
            validators = []

    if improver is None:
        try:
            improver = provider.get("improver")
        except DependencyError:
            pass  # Improver is optional

    # Create chain
    from sifaka.chain.chain import Chain
    return Chain(
        model=model_provider,
        validators=validators,
        improver=improver,
        max_attempts=max_attempts,
        name=name,
        description=description,
        config=config,
        **kwargs,
    )
```

### 3. Add Validation for Required Dependencies

#### 3.1 Validate Required Dependencies

```python
def create_simple_chain(
    model_provider: Optional[Any] = None,
    validators: Optional[List[Any]] = None,
    improver: Optional[Any] = None,
    max_attempts: int = 3,
    name: str = "simple_chain",
    description: str = "Simple chain for text generation and validation",
    config: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a simple chain.

    Args:
        model_provider: The model provider to use
        validators: The validators to use
        improver: The improver to use
        max_attempts: Maximum number of attempts
        name: The chain name
        description: The chain description
        config: Optional chain configuration
        **kwargs: Additional keyword arguments

    Returns:
        A simple chain instance

    Raises:
        ValueError: If required dependencies are missing
    """
    # Get dependency provider
    provider = DependencyProvider()

    # Resolve dependencies
    if model_provider is None:
        try:
            model_provider = provider.get("model_provider")
        except DependencyError:
            raise ValueError("Model provider is required")

    if validators is None:
        try:
            validators = provider.get("validators", [])
        except DependencyError:
            validators = []

    if improver is None:
        try:
            improver = provider.get("improver")
        except DependencyError:
            pass  # Improver is optional

    # Validate dependencies
    if model_provider is None:
        raise ValueError("Model provider is required")

    # Create chain
    from sifaka.chain.chain import Chain
    return Chain(
        model=model_provider,
        validators=validators,
        improver=improver,
        max_attempts=max_attempts,
        name=name,
        description=description,
        config=config,
        **kwargs,
    )
```

### 4. Use Type Annotations Consistently

#### 4.1 Add Type Annotations to Factory Functions

```python
from typing import Any, List, Optional, TypeVar, cast

from sifaka.interfaces import Chain, Model, Validator, Improver
from sifaka.utils.config import ChainConfig

T = TypeVar("T", bound=Chain)

def create_simple_chain(
    model_provider: Optional[Model] = None,
    validators: Optional[List[Validator]] = None,
    improver: Optional[Improver] = None,
    max_attempts: int = 3,
    name: str = "simple_chain",
    description: str = "Simple chain for text generation and validation",
    config: Optional[ChainConfig] = None,
    **kwargs: Any,
) -> T:
    """
    Create a simple chain.

    Args:
        model_provider: The model provider to use
        validators: The validators to use
        improver: The improver to use
        max_attempts: Maximum number of attempts
        name: The chain name
        description: The chain description
        config: Optional chain configuration
        **kwargs: Additional keyword arguments

    Returns:
        A simple chain instance

    Raises:
        ValueError: If required dependencies are missing
    """
    # Get dependency provider
    provider = DependencyProvider()

    # Resolve dependencies
    if model_provider is None:
        try:
            model_provider = cast(Model, provider.get("model_provider"))
        except DependencyError:
            raise ValueError("Model provider is required")

    if validators is None:
        try:
            validators = cast(List[Validator], provider.get("validators", []))
        except DependencyError:
            validators = []

    if improver is None:
        try:
            improver = cast(Optional[Improver], provider.get("improver"))
        except DependencyError:
            pass  # Improver is optional

    # Validate dependencies
    if model_provider is None:
        raise ValueError("Model provider is required")

    # Create chain
    from sifaka.chain.chain import Chain
    return cast(T, Chain(
        model=model_provider,
        validators=validators,
        improver=improver,
        max_attempts=max_attempts,
        name=name,
        description=description,
        config=config,
        **kwargs,
    ))
```

### 5. Remove Hard-Coded Dependencies

#### 5.1 Use Lazy Loading for Dependencies

```python
def create_model_provider(
    provider_type: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> Model:
    """
    Create a model provider.

    Args:
        provider_type: The provider type
        api_key: Optional API key
        model_name: Optional model name
        **kwargs: Additional keyword arguments

    Returns:
        A model provider instance

    Raises:
        ValueError: If the provider type is invalid
    """
    # Create provider based on type
    if provider_type == "openai":
        from sifaka.models.providers.openai import OpenAIProvider
        return OpenAIProvider(
            api_key=api_key,
            model_name=model_name or "gpt-3.5-turbo",
            **kwargs,
        )
    elif provider_type == "anthropic":
        from sifaka.models.providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            api_key=api_key,
            model_name=model_name or "claude-2",
            **kwargs,
        )
    elif provider_type == "gemini":
        from sifaka.models.providers.gemini import GeminiProvider
        return GeminiProvider(
            api_key=api_key,
            model_name=model_name or "gemini-pro",
            **kwargs,
        )
    elif provider_type == "mock":
        from sifaka.models.providers.mock import MockProvider
        return MockProvider(**kwargs)
    else:
        raise ValueError(f"Invalid provider type: {provider_type}")
```

## Specific File Changes

### sifaka/core/factories.py

```python
def create_chain(
    chain_type: str = "simple",
    model_provider: Optional[Model] = None,
    validators: Optional[List[Validator]] = None,
    improver: Optional[Improver] = None,
    max_attempts: int = 3,
    name: Optional[str] = None,
    description: Optional[str] = None,
    config: Optional[ChainConfig] = None,
    **kwargs: Any,
) -> Chain:
    """
    Create a chain with the specified configuration.

    This function creates a chain of the specified type with the given configuration,
    simplifying the creation of chains with common configurations.

    Args:
        chain_type: The type of chain to create ("simple" or "backoff")
        model_provider: The model provider to use
        validators: The validators to use
        improver: The improver to use
        max_attempts: Maximum number of attempts
        name: Optional chain name
        description: Optional chain description
        config: Optional chain configuration
        **kwargs: Additional keyword arguments

    Returns:
        A chain instance

    Raises:
        ValueError: If the chain type is invalid or required parameters are missing
    """
    # Validate chain type
    if chain_type not in ["simple", "backoff"]:
        raise ValueError(f"Invalid chain type: {chain_type}")

    # Set default name and description
    if name is None:
        name = f"{chain_type}_chain"
    if description is None:
        description = f"{chain_type.capitalize()} chain for text generation and validation"

    # Create chain based on type
    if chain_type == "simple":
        from sifaka.chain.factories import create_simple_chain
        return create_simple_chain(
            model_provider=model_provider,
            validators=validators,
            improver=improver,
            max_attempts=max_attempts,
            name=name,
            description=description,
            config=config,
            **kwargs,
        )
    else:  # chain_type == "backoff"
        from sifaka.chain.factories import create_backoff_chain
        return create_backoff_chain(
            model_provider=model_provider,
            validators=validators,
            improver=improver,
            max_attempts=max_attempts,
            name=name,
            description=description,
            config=config,
            **kwargs,
        )
```

## Success Criteria

1. Standardized parameter naming across factory functions
2. Dependency resolution in factory functions
3. Validation for required dependencies
4. Consistent type annotations
5. No hard-coded dependencies
6. Comprehensive documentation for factory functions
7. All factory functions follow standardized patterns
8. Tests validate proper factory function behavior
