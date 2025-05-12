"""
Factory functions for creating critics.

This module provides factory functions for creating critic instances with
standardized configuration and error handling.

## Overview
The module provides factory functions that simplify the creation of critic
instances with common configurations, handling parameter validation and
providing sensible defaults.

## Components
1. **create_critic**: Factory function for creating critic instances
2. **create_basic_critic**: Factory function for creating basic text critics

## Usage Examples
```python
from sifaka.critics.base.factories import create_critic, create_basic_critic
from sifaka.critics.base.implementation import Critic

# Create a critic with default settings
critic = create_critic(Critic)

# Create a critic with custom settings
critic = create_critic(
    Critic,
    name="custom_critic",
    description="A custom critic implementation",
    min_confidence=0.8,
    max_attempts=5,
    cache_size=200,
    priority=2,
    cost=1.5
)

# Create a basic critic
basic_critic = create_basic_critic(
    name="basic_critic",
    description="Basic text critic",
    min_confidence=0.8
)
```

## Error Handling
The functions handle:
- Invalid critic class
- Invalid configuration
- Missing required parameters
- Type validation
- Configuration validation
"""

from typing import Any, Optional, Type, TypeVar

from sifaka.utils.config.critics import CriticConfig
from sifaka.critics.base.abstract import BaseCritic
from sifaka.critics.base.implementation import Critic

# Default configuration values
DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CACHE_SIZE = 100

# Critic type variable
C = TypeVar("C", bound=BaseCritic)


def create_critic(
    critic_class: Type[C],
    name: str = "custom_critic",
    description: str = "Custom critic implementation",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    cache_size: int = DEFAULT_CACHE_SIZE,
    priority: int = 1,
    cost: float = 1.0,
    config: Optional[Optional[CriticConfig]] = None,
    **kwargs: Any,
) -> C:
    """
    Create a critic instance.

    This function provides a factory method for creating critic instances
    with standardized configuration and error handling.

    ## Overview
    The function:
    - Creates a critic instance with the specified configuration
    - Handles both direct config and parameter-based configuration
    - Provides default values for common parameters
    - Validates configuration before creating the critic

    ## Usage Examples
    ```python
    from sifaka.critics.base.factories import create_critic
    from sifaka.critics.base.implementation import Critic

    # Create a critic with default settings
    critic = create_critic(Critic)

    # Create a critic with custom settings
    critic = create_critic(
        Critic,
        name="custom_critic",
        description="A custom critic implementation",
        min_confidence=0.8,
        max_attempts=5,
        cache_size=200,
        priority=2,
        cost=1.5
    )

    # Create a critic with a config object
    from sifaka.utils.config.critics import CriticConfig
    config = CriticConfig(
        name="config_critic",
        description="Critic created with config object",
        min_confidence=0.9,
        max_attempts=3,
        cache_size=100,
        priority=1,
        cost=1.0
    )
    critic = create_critic(Critic, config=config)
    ```

    ## Error Handling
    The function handles:
    - Invalid critic class
    - Invalid configuration
    - Missing required parameters
    - Type validation
    - Configuration validation

    Args:
        critic_class: The critic class to instantiate
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache for memoization
        priority: Priority of the critic
        cost: Cost of running the critic
        config: Optional critic configuration
        **kwargs: Additional configuration parameters

    Returns:
        An instance of the specified critic class

    Raises:
        ValueError: If configuration is invalid
        TypeError: If critic_class is invalid
    """
    # Create config if not provided
    if config is None:
        config = CriticConfig(
            name=name,
            description=description,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            params=kwargs,
        )

    # Validate critic class
    if not issubclass(critic_class, BaseCritic):
        raise TypeError(f"critic_class must be a subclass of BaseCritic, got {critic_class}")

    # Create and return critic instance
    return critic_class(name, description, config)


def create_basic_critic(
    name: str = "basic_critic",
    description: str = "Basic text critic",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    **kwargs: Any,
) -> Critic:
    """
    Create a basic text critic.

    This function creates a basic critic instance with sensible defaults
    for common text processing tasks.

    ## Overview
    The function:
    - Creates a Critic instance with basic text processing capabilities
    - Provides default values for common parameters
    - Handles basic text validation and improvement
    - Includes simple critique functionality

    ## Usage Examples
    ```python
    from sifaka.critics.base.factories import create_basic_critic

    # Create a basic critic with default settings
    critic = create_basic_critic()

    # Create a basic critic with custom settings
    critic = create_basic_critic(
        name="custom_basic_critic",
        description="A custom basic critic",
        min_confidence=0.8,
        max_attempts=5
    )

    # Use the critic
    text = "This is a test text."
    result = (critic and critic.process(text)
    print(f"Score: {result.score:.2f}")
    print(f"Feedback: {result.message}")
    ```

    ## Error Handling
    The function handles:
    - Invalid configuration
    - Missing required parameters
    - Type validation
    - Configuration validation

    Args:
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        **kwargs: Additional configuration parameters

    Returns:
        A basic Critic instance

    Raises:
        ValueError: If configuration is invalid
    """
    return create_critic(
        Critic,
        name=name,
        description=description,
        min_confidence=min_confidence,
        max_attempts=max_attempts,
        **kwargs,
    )
