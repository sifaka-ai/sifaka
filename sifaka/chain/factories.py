"""
Chain Factories Module

This module provides factory functions for creating chains and components.
These factories simplify the creation of chains with sensible defaults.

## Factory Functions
1. **create_chain**: Creates a chain with the specified components
2. **create_simple_chain**: Creates a simple chain with the specified components
3. **create_backoff_chain**: Creates a chain with backoff retry strategy
4. **create_model_adapter**: Creates a model adapter for existing model providers
5. **create_validator_adapter**: Creates a validator adapter for existing rules
6. **create_improver_adapter**: Creates an improver adapter for existing critics

## Usage Examples
```python
from sifaka.chain.factories import create_simple_chain
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create components
model = OpenAIProvider("gpt-3.5-turbo")
validators = [create_length_rule(min_chars=10, max_chars=1000)]
critic = create_prompt_critic(
    llm_provider=model,
    system_prompt="You are an expert editor that improves text."
)

# Create chain using factory
chain = create_simple_chain(
    model=model,
    rules=validators,
    critic=critic,
    max_attempts=3
)

# Run chain
result = chain.run("Write a short story")
print(f"Output: {result.output}")
print(f"All validations passed: {result.all_passed}")
```
"""

from typing import Any, List, Optional

from .chain import Chain
from .interfaces import Model, Validator, Improver, Formatter
from .config import ChainConfig
from .adapters import ModelAdapter, ValidatorAdapter, ImproverAdapter, FormatterAdapter


def create_chain(
    model: Any,
    validators: List[Any] = None,
    improver: Optional[Any] = None,
    formatter: Optional[Any] = None,
    max_attempts: int = 3,
    config: Optional[ChainConfig] = None,
    name: str = "chain",
    description: str = "Sifaka chain for text generation and validation",
    **kwargs: Any,
) -> Chain:
    """
    Create a chain with the specified components.

    This factory function creates a chain with the specified components,
    automatically adapting them to the required interfaces if needed.
    It follows the standardized factory pattern used across Sifaka components.

    Args:
        model: The model to use for generation
        validators: The validators to use for validation
        improver: Optional improver for output improvement
        formatter: Optional formatter for result formatting
        max_attempts: Maximum number of generation attempts
        config: Chain configuration
        name: Chain name
        description: Chain description
        **kwargs: Additional keyword arguments for the chain

    Returns:
        A chain instance

    Raises:
        ValueError: If the parameters are invalid
        RuntimeError: If chain creation fails
    """
    try:
        # Adapt model if needed
        adapted_model = model if isinstance(model, Model) else ModelAdapter(model)

        # Adapt validators if needed
        adapted_validators = []
        if validators:
            for validator in validators:
                if isinstance(validator, Validator):
                    adapted_validators.append(validator)
                else:
                    adapted_validators.append(ValidatorAdapter(validator))

        # Adapt improver if needed
        adapted_improver = None
        if improver:
            if isinstance(improver, Improver):
                adapted_improver = improver
            else:
                adapted_improver = ImproverAdapter(improver)

        # Adapt formatter if needed
        adapted_formatter = None
        if formatter:
            if isinstance(formatter, Formatter):
                adapted_formatter = formatter
            else:
                adapted_formatter = FormatterAdapter(formatter)

        # Create chain
        chain = Chain(
            model=adapted_model,
            validators=adapted_validators,
            improver=adapted_improver,
            formatter=adapted_formatter,
            max_attempts=max_attempts,
            config=config,
            name=name,
            description=description,
            **kwargs,
        )

        # Chain is initialized in the constructor

        return chain
    except Exception as e:
        from ..utils.logging import get_logger

        logger = get_logger(__name__)
        logger.error(f"Error creating chain: {str(e)}")
        raise ValueError(f"Error creating chain: {str(e)}")


def create_simple_chain(
    model: Any = None,
    rules: List[Any] = None,
    critic: Optional[Any] = None,
    max_attempts: int = 3,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs: Any,
) -> Chain:
    """
    Create a simple chain with the specified components.

    This factory function creates a simple chain with the specified components,
    automatically adapting them to the required interfaces if needed.
    It follows the standardized factory pattern used across Sifaka components.
    It uses the dependency injection system to resolve dependencies if not explicitly provided.

    Args:
        model: The model to use for generation (injected if not provided)
        rules: The rules to use for validation (injected if not provided)
        critic: Optional critic for output improvement (injected if not provided)
        max_attempts: Maximum number of generation attempts
        session_id: Optional session ID for session-scoped dependencies
        request_id: Optional request ID for request-scoped dependencies
        **kwargs: Additional keyword arguments for the chain

    Returns:
        A chain instance

    Raises:
        ValueError: If the parameters are invalid
        RuntimeError: If chain creation fails
        DependencyError: If required dependencies cannot be resolved
    """
    try:
        # Resolve dependencies if not provided
        if model is None or rules is None or critic is None:
            from sifaka.core.dependency import DependencyProvider, DependencyError

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
                        from sifaka.interfaces.model import ModelProvider

                        model = provider.get_by_type(ModelProvider, None, session_id, request_id)
                    except (DependencyError, ImportError):
                        # This is a required dependency, so we need to raise an error
                        raise ValueError("Model provider is required for chain creation")

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

        # Create config if not provided
        config = kwargs.pop("config", None)
        if not config:
            config = ChainConfig(
                max_attempts=max_attempts,
                **{k: v for k, v in kwargs.items() if k not in ["session_id", "request_id"]},
            )

        # Create chain using the base factory function
        return create_chain(
            model=model,
            validators=rules,
            improver=critic,
            max_attempts=max_attempts,
            config=config,
            name=kwargs.get("name", "simple_chain"),
            description=kwargs.get(
                "description", "Simple chain for text generation and validation"
            ),
        )
    except Exception as e:
        from ..utils.logging import get_logger

        logger = get_logger(__name__)
        logger.error(f"Error creating simple chain: {str(e)}")
        raise ValueError(f"Error creating simple chain: {str(e)}")


def create_backoff_chain(
    model: Any = None,
    rules: List[Any] = None,
    critic: Optional[Any] = None,
    max_attempts: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **kwargs: Any,
) -> Chain:
    """
    Create a chain with backoff retry strategy.

    This factory function creates a chain with a backoff retry strategy,
    automatically adapting components to the required interfaces if needed.
    It follows the standardized factory pattern used across Sifaka components.
    It uses the dependency injection system to resolve dependencies if not explicitly provided.

    Args:
        model: The model to use for generation (injected if not provided)
        rules: The rules to use for validation (injected if not provided)
        critic: Optional critic for output improvement (injected if not provided)
        max_attempts: Maximum number of generation attempts
        initial_backoff: Initial backoff delay in seconds
        backoff_factor: Factor to multiply backoff by each attempt
        max_backoff: Maximum backoff delay in seconds
        session_id: Optional session ID for session-scoped dependencies
        request_id: Optional request ID for request-scoped dependencies
        **kwargs: Additional keyword arguments for the chain

    Returns:
        A chain instance

    Raises:
        ValueError: If the parameters are invalid
        RuntimeError: If chain creation fails
        DependencyError: If required dependencies cannot be resolved
    """
    try:
        # Resolve dependencies if not provided
        if model is None or rules is None or critic is None:
            from sifaka.core.dependency import DependencyProvider, DependencyError

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
                        from sifaka.interfaces.model import ModelProvider

                        model = provider.get_by_type(ModelProvider, None, session_id, request_id)
                    except (DependencyError, ImportError):
                        # This is a required dependency, so we need to raise an error
                        raise ValueError("Model provider is required for chain creation")

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

        # Create config if not provided
        config = kwargs.pop("config", None)
        if not config:
            from sifaka.chain.config import EngineConfig

            engine_config = EngineConfig(
                max_attempts=max_attempts,
                retry_delay=initial_backoff,
                backoff_factor=backoff_factor,
                max_retry_delay=max_backoff,
                jitter=kwargs.pop("jitter", True),
            )

            config = ChainConfig(
                max_attempts=max_attempts,
                params={"engine_config": engine_config},
                **{k: v for k, v in kwargs.items() if k not in ["session_id", "request_id"]},
            )

        # Create chain using the base factory function
        return create_chain(
            model=model,
            validators=rules,
            improver=critic,
            max_attempts=max_attempts,
            config=config,
            name=kwargs.get("name", "backoff_chain"),
            description=kwargs.get("description", "Chain with backoff retry strategy"),
            session_id=session_id,
            request_id=request_id,
        )
    except Exception as e:
        from ..utils.logging import get_logger

        logger = get_logger(__name__)
        logger.error(f"Error creating backoff chain: {str(e)}")
        raise ValueError(f"Error creating backoff chain: {str(e)}")
