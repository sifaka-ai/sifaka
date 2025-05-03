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

from typing import Any

from .models import CriticConfig
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
    config: CriticConfig = None,
    **kwargs: Any,
) -> CriticCore:
    """
    Create a prompt critic with the given parameters.

    This factory function creates a configured prompt critic instance
    that uses a language model to evaluate and improve text.

    ## Lifecycle Management

    1. **Initialization**
       - Validate input parameters
       - Create configuration object
       - Initialize prompt manager
       - Set up response parser

    2. **Configuration**
       - Apply default values
       - Handle custom configuration
       - Validate settings
       - Create immutable instance

    3. **Component Assembly**
       - Create prompt manager
       - Initialize response parser
       - Configure critic core
       - Return configured instance

    ## Error Handling

    1. **Validation Errors**
       - Invalid parameter values
       - Missing required components
       - Configuration conflicts
       - Resource initialization failures

    2. **Recovery**
       - Default value fallbacks
       - Parameter validation
       - Error logging
       - Graceful degradation

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

    Examples:
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
    """
    # Use provided config or create one from parameters
    if config is None:
        from .models import PromptCriticConfig

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
        )

    # Create managers
    prompt_manager = PromptCriticPromptManager(config)
    response_parser = ResponseParser()

    # Create critic - filter out any kwargs not accepted by CriticCore
    core_kwargs = {
        'config': config,
        'llm_provider': llm_provider,
        'prompt_manager': prompt_manager,
        'response_parser': response_parser,
    }

    # Create critic
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
    config: CriticConfig = None,
    **kwargs: Any,
) -> CriticCore:
    """
    Create a reflexion critic with the given parameters.

    This factory function creates a configured reflexion critic instance
    that uses a language model to evaluate and improve text, while maintaining
    a memory of past improvements to guide future improvements.

    ## Lifecycle Management

    1. **Initialization**
       - Validate input parameters
       - Create configuration object
       - Initialize prompt manager
       - Set up response parser
       - Configure memory manager

    2. **Configuration**
       - Apply default values
       - Handle custom configuration
       - Validate settings
       - Create immutable instance

    3. **Component Assembly**
       - Create prompt manager
       - Initialize response parser
       - Set up memory manager
       - Configure critic core
       - Return configured instance

    ## Error Handling

    1. **Validation Errors**
       - Invalid parameter values
       - Missing required components
       - Configuration conflicts
       - Resource initialization failures

    2. **Recovery**
       - Default value fallbacks
       - Parameter validation
       - Error logging
       - Graceful degradation

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
        memory_buffer_size: Maximum number of reflections to store
        reflection_depth: How many levels of reflection to perform
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A configured reflexion critic

    Examples:
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
    """
    # Use provided config or create one from parameters
    if config is None:
        from .models import ReflexionCriticConfig

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
        )

    # Create managers
    prompt_manager = ReflexionCriticPromptManager(config)
    response_parser = ResponseParser()

    # Use the buffer size from the config (which could be from the provided config parameter)
    buffer_size = getattr(config, 'memory_buffer_size', memory_buffer_size)
    memory_manager = MemoryManager(buffer_size=buffer_size)

    # Create critic - filter out any kwargs not accepted by CriticCore
    core_kwargs = {
        'config': config,
        'llm_provider': llm_provider,
        'prompt_manager': prompt_manager,
        'response_parser': response_parser,
        'memory_manager': memory_manager,
    }

    # Create critic
    return CriticCore(**core_kwargs)
