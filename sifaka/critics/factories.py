"""
Factory functions for creating critics.

This module provides factory functions for creating different types of critics,
including prompt-based critics and reflexion critics. These factories handle
the configuration and initialization of critics with their required components.

## Component Overview

1. **Factory Functions**
   - `create_prompt_critic`: Creates a prompt-based critic
   - `create_reflexion_critic`: Creates a reflexion critic

2. **Dependencies**
   - Language model providers
   - Prompt managers
   - Response parsers
   - Memory managers (for reflexion critics)

## Factory Lifecycle

1. **Initialization**
   - Validate input parameters
   - Create configuration objects
   - Initialize required managers
   - Configure critic components

2. **Configuration**
   - Set default values
   - Apply custom configurations
   - Validate settings
   - Create immutable instances

3. **Component Assembly**
   - Create prompt managers
   - Initialize response parsers
   - Set up memory managers
   - Configure critic core

## Error Handling

1. **Validation Errors**
   - Invalid parameter values
   - Missing required components
   - Configuration conflicts
   - Resource initialization failures

2. **Recovery Strategies**
   - Default value fallbacks
   - Parameter validation
   - Error logging
   - Graceful degradation

## Examples

Creating a prompt critic:

```python
from sifaka.critics.factories import create_prompt_critic
from sifaka.llm import OpenAIModel

# Create a language model provider
llm_provider = OpenAIModel(api_key="your-api-key")

# Create a prompt critic
critic = create_prompt_critic(
    llm_provider=llm_provider,
    name="my_critic",
    description="A custom prompt critic",
    min_confidence=0.8,
    system_prompt="You are an expert editor."
)

# Use the critic
result = critic.validate("Some text to validate")
```

Creating a reflexion critic:

```python
from sifaka.critics.factories import create_reflexion_critic
from sifaka.llm import OpenAIModel

# Create a language model provider
llm_provider = OpenAIModel(api_key="your-api-key")

# Create a reflexion critic
critic = create_reflexion_critic(
    llm_provider=llm_provider,
    name="my_reflexion_critic",
    description="A reflexion critic that learns from feedback",
    memory_buffer_size=10,
    reflection_depth=2
)

# Use the critic
result = critic.improve("Text to improve")
```

Using custom configurations:

```python
from sifaka.critics.models import PromptCriticConfig
from sifaka.critics.factories import create_prompt_critic
from sifaka.llm import OpenAIModel

# Create a custom configuration
config = PromptCriticConfig(
    name="custom_critic",
    description="A critic with custom settings",
    min_confidence=0.9,
    temperature=0.5
)

# Create a critic with the custom configuration
llm_provider = OpenAIModel(api_key="your-api-key")
critic = create_prompt_critic(
    llm_provider=llm_provider,
    config=config
)
```
"""

from typing import Any, Dict, Optional, Union

from .models import CriticConfig, PromptCriticConfig, ReflexionCriticConfig
from .core import CriticCore
from .managers.memory import MemoryManager
from .managers.prompt_factories import PromptCriticPromptManager, ReflexionCriticPromptManager
from .managers.response import ResponseParser
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_prompt_critic(
    llm_provider: Any,
    name: str = "prompt_critic",
    description: str = "Evaluates and improves text using language models",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = "You are an expert editor that improves text.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    config: Optional[Union[Dict[str, Any], CriticConfig]] = None,
    **kwargs: Any,
) -> CriticCore:
    """
    Create a prompt critic with the given parameters.

    This factory function creates a configured prompt critic instance
    that uses a language model to evaluate and improve text.

    Args:
        llm_provider: Language model provider to use
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache
        priority: Priority of the critic
        cost: Cost of using the critic
        system_prompt: System prompt for the model
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A configured prompt critic
    """
    # Create configuration
    if config is None:
        config = PromptCriticConfig(
            name=name,
            description=description,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    elif isinstance(config, dict):
        config = PromptCriticConfig(**{**config, **kwargs})

    # Create prompt manager
    prompt_manager = PromptCriticPromptManager(config=config)

    # Create response parser
    response_parser = ResponseParser()

    # Create core kwargs
    core_kwargs = {
        "config": config,
        "llm_provider": llm_provider,
        "prompt_manager": prompt_manager,
        "response_parser": response_parser,
    }

    # Create and return critic
    return CriticCore(**core_kwargs)


def create_reflexion_critic(
    llm_provider: Any,
    name: str = "reflexion_critic",
    description: str = "Improves text using reflections on past feedback",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = "You are an expert editor that learns from past feedback.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    memory_buffer_size: int = 5,
    reflection_depth: int = 1,
    config: Optional[Union[Dict[str, Any], CriticConfig]] = None,
    **kwargs: Any,
) -> CriticCore:
    """
    Create a reflexion critic with the given parameters.

    This factory function creates a configured reflexion critic instance
    that uses a language model and memory to evaluate and improve text.

    Args:
        llm_provider: Language model provider to use
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache
        priority: Priority of the critic
        cost: Cost of using the critic
        system_prompt: System prompt for the model
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        memory_buffer_size: Size of the memory buffer
        reflection_depth: Depth of reflection
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A configured reflexion critic
    """
    # Create configuration
    if config is None:
        config = ReflexionCriticConfig(
            name=name,
            description=description,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            memory_buffer_size=memory_buffer_size,
            reflection_depth=reflection_depth,
            **kwargs,
        )
    elif isinstance(config, dict):
        config = ReflexionCriticConfig(**{**config, **kwargs})

    # Create prompt manager
    prompt_manager = ReflexionCriticPromptManager(config=config)

    # Create response parser
    response_parser = ResponseParser()

    # Create memory manager
    memory_manager = MemoryManager(buffer_size=memory_buffer_size)

    # Create core kwargs
    core_kwargs = {
        "config": config,
        "llm_provider": llm_provider,
        "prompt_manager": prompt_manager,
        "response_parser": response_parser,
        "memory_manager": memory_manager,
    }

    # Create and return critic
    return CriticCore(**core_kwargs)
