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

from sifaka.utils.error_handling import ConfigurationError

if TYPE_CHECKING:
    from sifaka.storage.protocol import Storage

# Check if PydanticAI is available
try:
    from pydantic_ai import Agent

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None

# Import components with availability checks
__all__ = []

if PYDANTIC_AI_AVAILABLE:
    from sifaka.agents.chain import PydanticAIChain
    from sifaka.models.pydantic_ai import PydanticAIModel

    __all__.extend(
        ["PydanticAIChain", "PydanticAIModel", "create_pydantic_chain", "create_agent_model"]
    )


def create_pydantic_chain(
    agent: "Agent",
    validators: Optional[List] = None,
    critics: Optional[List] = None,
    storage: Optional["Storage"] = None,
    **kwargs,
) -> "PydanticAIChain":
    """Factory function to create a PydanticAI chain with Sifaka components.

    Args:
        agent: The PydanticAI agent to use for generation.
        validators: Optional list of Sifaka validators.
        critics: Optional list of Sifaka critics.
        storage: Optional storage backend for thoughts.
        **kwargs: Additional arguments passed to PydanticAIChain.

    Returns:
        A PydanticAIChain instance.

    Raises:
        ConfigurationError: If PydanticAI is not available.
    """
    if not PYDANTIC_AI_AVAILABLE:
        raise ConfigurationError(
            "PydanticAI is not available. Please install it with: pip install pydantic-ai",
            suggestions=[
                "Install PydanticAI: pip install pydantic-ai",
                "Or use uv: uv add pydantic-ai",
            ],
        )

    from sifaka.agents.chain import PydanticAIChain

    return PydanticAIChain(
        agent=agent,
        storage=storage,
        validators=validators or [],
        critics=critics or [],
        **kwargs,
    )


def create_agent_model(agent: "Agent", **kwargs) -> "PydanticAIModel":
    """Factory function to create a PydanticAI model adapter.

    Args:
        agent: The PydanticAI agent to wrap.
        **kwargs: Additional arguments passed to PydanticAIModel.

    Returns:
        A PydanticAIModel instance.

    Raises:
        ConfigurationError: If PydanticAI is not available.
    """
    if not PYDANTIC_AI_AVAILABLE:
        raise ConfigurationError(
            "PydanticAI is not available. Please install it with: pip install pydantic-ai",
            suggestions=[
                "Install PydanticAI: pip install pydantic-ai",
                "Or use uv: uv add pydantic-ai",
            ],
        )

    from sifaka.models.pydantic_ai import PydanticAIModel

    return PydanticAIModel(agent=agent, **kwargs)
