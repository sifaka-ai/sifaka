"""PydanticAI integration for Sifaka.

This module provides integration between Sifaka and PydanticAI, enabling
hybrid workflows that combine PydanticAI's agent capabilities with Sifaka's
evaluation and improvement framework.

Example:
    ```python
    from pydantic_ai import Agent
    from sifaka.agents import create_pydantic_chain
    from sifaka.validators import LengthValidator
    from sifaka.critics import ReflexionCritic
    from sifaka.models import create_model

    # Create PydanticAI agent
    agent = Agent('openai:gpt-4o', system_prompt='You are a helpful assistant.')

    # Create Sifaka components
    validator = LengthValidator(min_length=100, max_length=500)
    critic = ReflexionCritic(model=create_model("openai:gpt-3.5-turbo"))

    # Create hybrid chain
    chain = create_pydantic_chain(
        agent=agent,
        validators=[validator],
        critics=[critic]
    )

    # Run the chain
    result = chain.run("Write a story about AI")
    print(f"Generated: {result.text}")
    ```
"""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sifaka.storage.protocol import Storage

# PydanticAI is a required dependency
from pydantic_ai import Agent

from sifaka.agents.chain import PydanticAIChain
from sifaka.agents.tools import (
    create_criticism_tool,
    create_self_correcting_agent,
    create_validation_tool,
)
from sifaka.models.pydantic_ai import PydanticAIModel

__all__ = [
    "PydanticAIChain",
    "PydanticAIModel",
    "create_pydantic_chain",
    "create_agent_model",
    "create_validation_tool",
    "create_criticism_tool",
    "create_self_correcting_agent",
]


def create_pydantic_chain(
    agent: "Agent",
    validators: Optional[List] = None,
    critics: Optional[List] = None,
    model_retrievers: Optional[List] = None,
    critic_retrievers: Optional[List] = None,
    max_improvement_iterations: int = 2,
    retries: Optional[int] = None,  # Alias for max_improvement_iterations
    always_apply_critics: bool = False,
    analytics_storage: Optional["Storage"] = None,
    storage: Optional["Storage"] = None,  # Alias for analytics_storage
    **kwargs,
) -> "PydanticAIChain":
    """Factory function to create a PydanticAI chain with Sifaka components.

    Args:
        agent: The PydanticAI agent to use for generation.
        validators: Optional list of Sifaka validators to always run if provided.
        critics: Optional list of Sifaka critics to always run if provided.
        model_retrievers: Optional list of retrievers for pre-generation context injection.
        critic_retrievers: Optional list of retrievers for pre-critic context injection.
        max_improvement_iterations: Maximum number of improvement iterations (default: 2).
        always_apply_critics: Whether to always apply critics even on first success (default: False).
        analytics_storage: Optional storage backend for analytics/debugging only.
                          PydanticAI conversation history is the primary memory.
        storage: Alias for analytics_storage (for backward compatibility).
        **kwargs: Additional arguments passed to PydanticAIChain.

    Returns:
        A PydanticAIChain instance.

    """

    from sifaka.agents.chain import PydanticAIChain

    # Handle storage parameter alias - prefer analytics_storage if both are provided
    # Use explicit None check instead of 'or' to avoid issues with empty storage objects
    # that evaluate to False due to __len__ returning 0
    final_storage = analytics_storage if analytics_storage is not None else storage

    return PydanticAIChain(
        agent=agent,
        validators=validators or [],
        critics=critics or [],
        model_retrievers=model_retrievers or [],
        critic_retrievers=critic_retrievers or [],
        max_improvement_iterations=max_improvement_iterations,
        always_apply_critics=always_apply_critics,
        analytics_storage=final_storage,
        **kwargs,
    )


def create_agent_model(agent: "Agent", **kwargs) -> "PydanticAIModel":
    """Factory function to create a PydanticAI model adapter.

    Args:
        agent: The PydanticAI agent to wrap.
        **kwargs: Additional arguments passed to PydanticAIModel.

    Returns:
        A PydanticAIModel instance.

    """

    from sifaka.models.pydantic_ai import PydanticAIModel

    return PydanticAIModel(agent=agent, **kwargs)
