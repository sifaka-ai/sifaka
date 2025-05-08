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

from typing import Any, Dict, List, Optional, Union

from .models import (
    CriticConfig,
    PromptCriticConfig,
    ReflexionCriticConfig,
    SelfRefineCriticConfig,
    SelfRAGCriticConfig,
    ConstitutionalCriticConfig,
)
from .lac import (
    FeedbackCriticConfig,
    ValueCriticConfig,
    LACCriticConfig,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
    DEFAULT_VALUE_PROMPT_TEMPLATE,
)
from .core import CriticCore
from .base import CompositionCritic, create_composition_critic
from .managers.prompt_factories import PromptCriticPromptManager, ReflexionCriticPromptManager
from .managers.response import ResponseParser
from .implementations.prompt_implementation import PromptCriticImplementation
from .implementations.reflexion_implementation import ReflexionCriticImplementation
from .implementations.self_refine_implementation import SelfRefineCriticImplementation
from .implementations.self_rag_implementation import SelfRAGCriticImplementation
from .implementations.constitutional_implementation import ConstitutionalCriticImplementation
from .implementations.lac_implementation import (
    FeedbackCriticImplementation,
    ValueCriticImplementation,
    LACCriticImplementation,
)
from ..utils.logging import get_logger
from ..utils.config import standardize_critic_config
from ..retrieval import Retriever

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
    # Create configuration using the standardization utility
    config = standardize_critic_config(
        config=config,
        config_class=PromptCriticConfig,
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
) -> CompositionCritic:
    """
    Create a reflexion critic with the given parameters.

    This factory function creates a configured reflexion critic instance
    that uses a language model and memory to evaluate and improve text.
    It follows the composition over inheritance pattern.

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
        A configured reflexion critic using composition over inheritance
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

    # Create implementation
    implementation = ReflexionCriticImplementation(
        config=config,
        llm_provider=llm_provider,
        prompt_factory=prompt_manager,
    )

    # Create and return critic using composition
    return create_composition_critic(
        name=name,
        description=description,
        implementation=implementation,
        config=config,
    )


def create_self_refine_critic(
    llm_provider: Any,
    name: str = "self_refine_critic",
    description: str = "Improves text through iterative self-critique and revision",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = "You are an expert at critiquing and revising content.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    max_iterations: int = 3,
    critique_prompt_template: Optional[str] = None,
    revision_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], SelfRefineCriticConfig]] = None,
    **kwargs: Any,
) -> CompositionCritic:
    """
    Create a self-refine critic with the given parameters.

    This factory function creates a configured self-refine critic instance
    that uses a language model to iteratively critique and revise text outputs.
    It follows the composition over inheritance pattern.

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
        max_iterations: Maximum number of refinement iterations
        critique_prompt_template: Optional custom template for critique prompts
        revision_prompt_template: Optional custom template for revision prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A configured self-refine critic using composition over inheritance
    """
    # Create configuration
    if config is None:
        config_dict = {
            "name": name,
            "description": description,
            "min_confidence": min_confidence,
            "max_attempts": max_attempts,
            "cache_size": cache_size,
            "priority": priority,
            "cost": cost,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_iterations": max_iterations,
            **kwargs,  # Include additional kwargs in the config
        }

        if critique_prompt_template:
            config_dict["critique_prompt_template"] = critique_prompt_template

        if revision_prompt_template:
            config_dict["revision_prompt_template"] = revision_prompt_template

        config = SelfRefineCriticConfig(**config_dict)
    elif isinstance(config, dict):
        config = SelfRefineCriticConfig(**{**config, **kwargs})

    # Create prompt manager
    prompt_manager = PromptCriticPromptManager(config=config)

    # Create implementation
    implementation = SelfRefineCriticImplementation(
        config=config,
        llm_provider=llm_provider,
        prompt_factory=prompt_manager,
    )

    # Create and return critic using composition
    return create_composition_critic(
        name=name,
        description=description,
        implementation=implementation,
        config=config,
    )


def create_self_rag_critic(
    llm_provider: Any,
    retriever: Retriever,
    name: str = "self_rag_critic",
    description: str = "Improves text through self-reflective retrieval-augmented generation",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = "You are an expert at deciding when to retrieve information and reflecting on its relevance.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    retrieval_threshold: float = 0.5,
    retrieval_prompt_template: Optional[str] = None,
    generation_prompt_template: Optional[str] = None,
    reflection_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], SelfRAGCriticConfig]] = None,
    **kwargs: Any,
) -> CompositionCritic:
    """
    Create a Self-RAG critic with the given parameters.

    This factory function creates a configured Self-RAG critic instance
    that uses a language model and retrieval to evaluate and improve text.
    It follows the composition over inheritance pattern.

    Args:
        llm_provider: Language model provider to use
        retriever: Retriever to use for information retrieval
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
        retrieval_threshold: Threshold for retrieval confidence
        retrieval_prompt_template: Optional custom template for retrieval prompts
        generation_prompt_template: Optional custom template for generation prompts
        reflection_prompt_template: Optional custom template for reflection prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A configured Self-RAG critic using composition over inheritance

    Examples:
        ```python
        from sifaka.critics.factories import create_self_rag_critic
        from sifaka.models.providers import OpenAIProvider
        from sifaka.retrieval import create_simple_retriever

        # Create a language model provider
        provider = OpenAIProvider(api_key="your-api-key")

        # Create a retriever
        documents = {
            "health_insurance": "To file a claim for health reimbursement, follow these steps: "
                               "1. Complete the claim form with your personal and policy information. "
                               "2. Attach all original receipts and medical documentation. "
                               "3. Make copies of all documents for your records.",
            "travel_insurance": "For travel insurance claims, you need to provide: "
                               "1. Proof of travel (boarding passes, itinerary). "
                               "2. Incident report or documentation of the event. "
                               "3. Original receipts for expenses being claimed."
        }
        retriever = create_simple_retriever(documents=documents)

        # Create a Self-RAG critic
        critic = create_self_rag_critic(
            llm_provider=provider,
            retriever=retriever,
            name="insurance_rag_critic",
            description="A critic for insurance-related queries",
            system_prompt="You are an expert at retrieving and using insurance information."
        )

        # Use the critic
        text = "What are the steps to file a health insurance claim?"
        is_valid = critic.validate(text)
        improved = critic.improve(text)
        feedback = critic.critique(text)
        ```
    """
    # Try to use standardize_critic_config if available
    try:
        # Create configuration using the standardization utility
        critic_config = standardize_critic_config(
            config_class=SelfRAGCriticConfig,
            config=config,
            name=name,
            description=description,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            retrieval_threshold=retrieval_threshold,
            retrieval_prompt_template=retrieval_prompt_template,
            generation_prompt_template=generation_prompt_template,
            reflection_prompt_template=reflection_prompt_template,
            **kwargs,
        )
    except Exception:
        # If standardize_critic_config fails, create config manually
        if config is None:
            # Create new config
            config_dict = {
                "name": name,
                "description": description,
                "min_confidence": min_confidence,
                "max_attempts": max_attempts,
                "cache_size": cache_size,
                "priority": priority,
                "cost": cost,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "retrieval_threshold": retrieval_threshold,
                **kwargs,  # Include additional kwargs in the config
            }

            # Add optional parameters if provided
            if retrieval_prompt_template:
                config_dict["retrieval_prompt_template"] = retrieval_prompt_template
            if generation_prompt_template:
                config_dict["generation_prompt_template"] = generation_prompt_template
            if reflection_prompt_template:
                config_dict["reflection_prompt_template"] = reflection_prompt_template

            critic_config = SelfRAGCriticConfig(**config_dict)
        elif isinstance(config, dict):
            # Convert dict to config
            critic_config = SelfRAGCriticConfig(**{**config, **kwargs})
        else:
            # Use provided config
            critic_config = config

    # Create prompt manager
    prompt_manager = PromptCriticPromptManager(config=critic_config)

    # Create implementation
    implementation = SelfRAGCriticImplementation(
        config=critic_config,
        llm_provider=llm_provider,
        retriever=retriever,
        prompt_factory=prompt_manager,
    )

    # Create and return critic using composition
    return create_composition_critic(
        name=name,
        description=description,
        implementation=implementation,
        config=critic_config,
    )


def create_feedback_critic(
    llm_provider: Any,
    name: str = "feedback_critic",
    description: str = "Provides natural language feedback for text",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    feedback_prompt_template: str = DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
    config: Optional[Union[Dict[str, Any], FeedbackCriticConfig]] = None,
    **kwargs: Any,
) -> CompositionCritic:
    """
    Create a feedback critic with the given parameters.

    This factory function creates a configured feedback critic instance
    that provides natural language feedback for text.

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
        feedback_prompt_template: Template for feedback prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A configured feedback critic using composition over inheritance

    Examples:
        ```python
        from sifaka.critics.factories import create_feedback_critic
        from sifaka.models.providers import OpenAIProvider

        # Create a language model provider
        provider = OpenAIProvider(api_key="your-api-key")

        # Create a feedback critic
        critic = create_feedback_critic(
            llm_provider=provider,
            name="my_feedback_critic",
            description="A custom feedback critic",
            system_prompt="You are an expert at providing constructive feedback.",
            temperature=0.7,
            max_tokens=1000
        )

        # Use the critic
        task = "Summarize the causes of World War I in 3 bullet points."
        response = "World War I was caused by nationalism, militarism, and alliances."
        feedback = critic.critique(response, {"task": task})
        print(f"Feedback: {feedback['feedback']}")
        ```
    """
    # Create configuration
    if config is None:
        config_dict = {
            "name": name,
            "description": description,
            "min_confidence": min_confidence,
            "max_attempts": max_attempts,
            "cache_size": cache_size,
            "priority": priority,
            "cost": cost,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "feedback_prompt_template": feedback_prompt_template,
            **kwargs,
        }
        config = FeedbackCriticConfig(**config_dict)
    elif isinstance(config, dict):
        config = FeedbackCriticConfig(**{**config, **kwargs})

    # Create implementation
    implementation = FeedbackCriticImplementation(
        config=config,
        llm_provider=llm_provider,
    )

    # Create and return critic using composition
    return create_composition_critic(
        name=name,
        description=description,
        implementation=implementation,
        config=config,
    )


def create_value_critic(
    llm_provider: Any,
    name: str = "value_critic",
    description: str = "Estimates numeric values for text",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.3,
    max_tokens: int = 100,
    value_prompt_template: str = DEFAULT_VALUE_PROMPT_TEMPLATE,
    config: Optional[Union[Dict[str, Any], ValueCriticConfig]] = None,
    **kwargs: Any,
) -> CompositionCritic:
    """
    Create a value critic with the given parameters.

    This factory function creates a configured value critic instance
    that estimates numeric values for text.

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
        value_prompt_template: Template for value prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A configured value critic using composition over inheritance

    Examples:
        ```python
        from sifaka.critics.factories import create_value_critic
        from sifaka.models.providers import OpenAIProvider

        # Create a language model provider
        provider = OpenAIProvider(api_key="your-api-key")

        # Create a value critic
        critic = create_value_critic(
            llm_provider=provider,
            name="my_value_critic",
            description="A custom value critic",
            system_prompt="You are an expert at estimating the quality of responses.",
            temperature=0.3,
            max_tokens=100
        )

        # Use the critic
        task = "Summarize the causes of World War I in 3 bullet points."
        response = "World War I was caused by nationalism, militarism, and alliances."
        value = critic.critique(response, {"task": task})
        print(f"Value: {value['value']}")
        ```
    """
    # Create configuration
    if config is None:
        config_dict = {
            "name": name,
            "description": description,
            "min_confidence": min_confidence,
            "max_attempts": max_attempts,
            "cache_size": cache_size,
            "priority": priority,
            "cost": cost,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "value_prompt_template": value_prompt_template,
            **kwargs,
        }
        config = ValueCriticConfig(**config_dict)
    elif isinstance(config, dict):
        config = ValueCriticConfig(**{**config, **kwargs})

    # Create implementation
    implementation = ValueCriticImplementation(
        config=config,
        llm_provider=llm_provider,
    )

    # Create and return critic using composition
    return create_composition_critic(
        name=name,
        description=description,
        implementation=implementation,
        config=config,
    )


def create_lac_critic(
    llm_provider: Any,
    name: str = "lac_critic",
    description: str = "Combines language feedback and value scoring",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    feedback_prompt_template: str = DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
    value_prompt_template: str = DEFAULT_VALUE_PROMPT_TEMPLATE,
    config: Optional[Union[Dict[str, Any], LACCriticConfig]] = None,
    **kwargs: Any,
) -> CompositionCritic:
    """
    Create a LAC critic with the given parameters.

    This factory function creates a configured LAC critic instance
    that combines language feedback and value scoring.

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
        feedback_prompt_template: Template for feedback prompts
        value_prompt_template: Template for value prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A configured LAC critic using composition over inheritance

    Examples:
        ```python
        from sifaka.critics.factories import create_lac_critic
        from sifaka.models.providers import OpenAIProvider

        # Create a language model provider
        provider = OpenAIProvider(api_key="your-api-key")

        # Create a LAC critic
        critic = create_lac_critic(
            llm_provider=provider,
            name="my_lac_critic",
            description="A custom LAC critic",
            system_prompt="You are an expert at evaluating and improving text.",
            temperature=0.7,
            max_tokens=1000
        )

        # Use the critic
        task = "Summarize the causes of World War I in 3 bullet points."
        response = "World War I was caused by nationalism, militarism, and alliances."
        result = critic.critique(response, {"task": task})
        print(f"Feedback: {result['feedback']}")
        print(f"Value: {result['value']}")
        ```
    """
    # Create configuration
    if config is None:
        config_dict = {
            "name": name,
            "description": description,
            "min_confidence": min_confidence,
            "max_attempts": max_attempts,
            "cache_size": cache_size,
            "priority": priority,
            "cost": cost,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "feedback_prompt_template": feedback_prompt_template,
            "value_prompt_template": value_prompt_template,
            **kwargs,
        }
        config = LACCriticConfig(**config_dict)
    elif isinstance(config, dict):
        config = LACCriticConfig(**{**config, **kwargs})

    # Create implementation
    implementation = LACCriticImplementation(
        config=config,
        llm_provider=llm_provider,
    )

    # Create and return critic using composition
    return create_composition_critic(
        name=name,
        description=description,
        implementation=implementation,
        config=config,
    )


def create_constitutional_critic(
    llm_provider: Any,
    principles: List[str],
    name: str = "constitutional_critic",
    description: str = "Evaluates responses against principles",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    system_prompt: str = "You are an expert at evaluating content against principles.",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    critique_prompt_template: Optional[str] = None,
    improvement_prompt_template: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], ConstitutionalCriticConfig]] = None,
    **kwargs: Any,
) -> CompositionCritic:
    """
    Create a constitutional critic with the given parameters.

    This factory function creates a configured constitutional critic instance
    that evaluates responses against a set of principles. It follows the
    composition over inheritance pattern.

    Args:
        llm_provider: Language model provider to use
        principles: List of principles to evaluate responses against
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
        critique_prompt_template: Optional custom template for critique prompts
        improvement_prompt_template: Optional custom template for improvement prompts
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        A configured constitutional critic using composition over inheritance

    Raises:
        ValueError: If principles is empty or invalid
        TypeError: If llm_provider is not a valid language model
    """
    # Create configuration
    if config is None:
        config_dict = {
            "name": name,
            "description": description,
            "min_confidence": min_confidence,
            "max_attempts": max_attempts,
            "cache_size": cache_size,
            "priority": priority,
            "cost": cost,
            "principles": principles,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "params": {},
            **kwargs,
        }

        if critique_prompt_template:
            config_dict["params"]["critique_prompt_template"] = critique_prompt_template

        if improvement_prompt_template:
            config_dict["params"]["improvement_prompt_template"] = improvement_prompt_template

        config = ConstitutionalCriticConfig(**config_dict)
    elif isinstance(config, dict):
        # Ensure principles are included in the config
        if "principles" not in config and principles:
            config["principles"] = principles
        config = ConstitutionalCriticConfig(**config)
    elif not isinstance(config, ConstitutionalCriticConfig):
        raise TypeError("config must be a ConstitutionalCriticConfig or dict")

    # Create implementation
    implementation = ConstitutionalCriticImplementation(
        config=config,
        llm_provider=llm_provider,
    )

    # Create and return critic
    return create_composition_critic(
        name=name,
        description=description,
        implementation=implementation,
        config=config,
    )
