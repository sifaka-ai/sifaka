"""
Chain Factory Module

Factory functions for creating different types of chains.

## Overview
This module provides factory functions for creating different types of chains,
including simple chains and chains with exponential backoff retry strategies.

## Components
1. **create_simple_chain**: Creates a basic chain with simple retry strategy
2. **create_backoff_chain**: Creates a chain with exponential backoff retry

## Usage Examples
```python
from sifaka.chain import create_simple_chain, create_backoff_chain
from sifaka.models import create_openai_chat_provider
from sifaka.rules import create_length_rule, create_toxicity_rule
from sifaka.critics import create_prompt_critic

# Create model provider
model_provider = create_openai_chat_provider(
    model_name="gpt-3.5-turbo",
    api_key="your-api-key"
)

# Create rules
rules = [
    create_length_rule(min_length=10, max_length=1000),
    create_toxicity_rule(threshold=0.7)
]

# Create critic
critic = create_prompt_critic(
    llm_provider=model_provider,
    system_prompt="You are an expert editor that improves text."
)

# Create simple chain
simple_chain = create_simple_chain(
    model=model_provider,
    rules=rules,
    critic=critic,
    max_attempts=3
)

# Create chain with backoff
backoff_chain = create_backoff_chain(
    model=model_provider,
    rules=rules,
    critic=critic,
    max_attempts=3,
    initial_backoff=1.0,
    backoff_factor=2.0,
    max_backoff=60.0
)

# Run the chains
simple_result = simple_chain.run("Write a short story about a robot learning to paint.")
backoff_result = backoff_chain.run("Write a poem about autumn.")
```

## Error Handling
- ValueError: Raised when validation fails after max attempts
- ChainError: Raised when chain execution fails
- ValidationError: Raised when validation fails
- CriticError: Raised when critic refinement fails
- ModelError: Raised when model generation fails

## Configuration
- model: The model provider for text generation
- rules: List of rules to validate outputs against
- critic: Optional critic for improving outputs
- max_attempts: Maximum number of validation attempts
- initial_backoff: Initial backoff time in seconds (backoff chain only)
- backoff_factor: Factor to multiply backoff by each attempt (backoff chain only)
- max_backoff: Maximum backoff time in seconds (backoff chain only)
"""

from typing import Generic, List, Optional, TypeVar

from ..critics import CriticCore
from ..models.base import ModelProvider
from ..rules import Rule
from .orchestrator import ChainOrchestrator
from .formatters.result import ResultFormatter
from .managers.prompt import PromptManager
from .managers.validation import ValidationManager
from .strategies.retry import BackoffRetryStrategy, SimpleRetryStrategy
from ..utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


def create_simple_chain(
    model: ModelProvider,
    rules: List[Rule],
    critic: Optional[CriticCore] = None,
    max_attempts: int = 3,
) -> ChainOrchestrator[OutputType]:
    """
    Create a simple chain with the given parameters.

    Detailed description of what the function does, including:
    - Creates a basic chain with a simple retry strategy
    - Uses the provided model provider, rules, and optional critic
    - Configures standard validation and prompt managers

    Args:
        model (ModelProvider): The model provider to use
        rules (List[Rule]): The rules to validate against
        critic (Optional[CriticCore]): Optional critic to use
        max_attempts (int): Maximum number of attempts

    Returns:
        ChainOrchestrator[OutputType]: A configured chain orchestrator

    Raises:
        ValueError: When validation fails after max attempts
        ChainError: When chain execution fails
        ValidationError: When validation fails
        CriticError: When critic refinement fails
        ModelError: When model generation fails

    Example:
        ```python
        from sifaka.chain import create_simple_chain
        from sifaka.models import create_openai_chat_provider
        from sifaka.rules import create_length_rule, create_toxicity_rule

        # Create model provider
        model_provider = create_openai_chat_provider(
            model_name="gpt-3.5-turbo",
            api_key="your-api-key"
        )

        # Create rules
        rules = [
            create_length_rule(min_length=10, max_length=1000),
            create_toxicity_rule(threshold=0.7)
        ]

        # Create simple chain
        chain = create_simple_chain(
            model=model_provider,
            rules=rules,
            max_attempts=3
        )

        # Run the chain
        result = chain.run("Write a short story about a robot learning to paint.")

        # Check the result
        print(f"Output: {result.output}")
        print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
        ```
    """
    return ChainOrchestrator[OutputType](
        model=model,
        rules=rules,
        critic=critic,
        max_attempts=max_attempts,
    )


def create_backoff_chain(
    model: ModelProvider,
    rules: List[Rule],
    critic: Optional[CriticCore] = None,
    max_attempts: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
) -> ChainOrchestrator[OutputType]:
    """
    Create a chain with exponential backoff retry strategy.

    Detailed description of what the function does, including:
    - Creates a chain with an exponential backoff retry strategy
    - Increases wait time between retry attempts to handle rate limits
    - Uses the provided model provider, rules, and optional critic

    Args:
        model (ModelProvider): The model provider to use
        rules (List[Rule]): The rules to validate against
        critic (Optional[CriticCore]): Optional critic to use
        max_attempts (int): Maximum number of attempts
        initial_backoff (float): Initial backoff time in seconds
        backoff_factor (float): Factor to multiply backoff by each attempt
        max_backoff (float): Maximum backoff time in seconds

    Returns:
        ChainOrchestrator[OutputType]: A configured chain orchestrator

    Raises:
        ValueError: When validation fails after max attempts
        ChainError: When chain execution fails
        ValidationError: When validation fails
        CriticError: When critic refinement fails
        ModelError: When model generation fails

    Example:
        ```python
        from sifaka.chain import create_backoff_chain
        from sifaka.models import create_openai_chat_provider
        from sifaka.rules import create_length_rule, create_toxicity_rule

        # Create model provider
        model_provider = create_openai_chat_provider(
            model_name="gpt-3.5-turbo",
            api_key="your-api-key"
        )

        # Create rules
        rules = [
            create_length_rule(min_length=10, max_length=1000),
            create_toxicity_rule(threshold=0.7)
        ]

        # Create chain with backoff
        chain = create_backoff_chain(
            model=model_provider,
            rules=rules,
            max_attempts=3,
            initial_backoff=1.0,
            backoff_factor=2.0,
            max_backoff=60.0
        )

        # Run the chain
        result = chain.run("Write a poem about autumn.")

        # Check the result
        print(f"Output: {result.output}")
        print(f"All rules passed: {all(r.passed for r in result.rule_results)}")
        ```
    """
    # Create components
    validation_manager = ValidationManager[OutputType](rules)
    prompt_manager = PromptManager()
    retry_strategy = BackoffRetryStrategy[OutputType](
        max_attempts=max_attempts,
        initial_backoff=initial_backoff,
        backoff_factor=backoff_factor,
        max_backoff=max_backoff,
    )
    result_formatter = ResultFormatter[OutputType]()

    # Create core chain
    core = ChainCore[OutputType](
        model=model,
        validation_manager=validation_manager,
        prompt_manager=prompt_manager,
        retry_strategy=retry_strategy,
        result_formatter=result_formatter,
        critic=critic,
    )

    return ChainOrchestrator[OutputType](
        model=model,
        rules=rules,
        critic=critic,
        max_attempts=max_attempts,
    )
