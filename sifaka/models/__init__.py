"""Models for Sifaka.

This module provides the unified model creation interface using PydanticAI
as the backend for all language model providers.

After the PydanticAI migration, all model creation goes through the
create_model() factory function which uses PydanticAI agents internally.

Example:
    ```python
    from sifaka.models import create_model

    # Create models using PydanticAI backend
    model = create_model("openai:gpt-4")
    model = create_model("anthropic:claude-3-sonnet")
    model = create_model("google:gemini-1.5-flash")
    model = create_model("ollama:llama2")

    # Use in critics (legacy Sifaka model interface)
    response = model.generate("Write a story about AI")
    print(response)
    ```

For new code, prefer using PydanticAI agents directly:
    ```python
    from pydantic_ai import Agent
    from sifaka.agents import create_pydantic_chain

    # Modern approach - use PydanticAI agents directly
    agent = Agent("openai:gpt-4", system_prompt="You are helpful.")
    chain = create_pydantic_chain(agent=agent, validators=[], critics=[])
    ```
"""

from sifaka.models.base import create_model
from sifaka.models.critic_results import (
    CriticResult,
    CritiqueFeedback,
    ImprovementSuggestion,
    ViolationReport,
    ConfidenceScore,
    SeverityLevel,
)
from sifaka.models.pydantic_ai import PydanticAIModel, create_pydantic_ai_model

# Only export the unified factory function and PydanticAI adapter
__all__ = ["create_model"]

# PydanticAI adapter (for backward compatibility with critics)
__all__.extend(["PydanticAIModel", "create_pydantic_ai_model"])

# Critic result models
__all__.extend(
    [
        "CriticResult",
        "CritiqueFeedback",
        "ImprovementSuggestion",
        "ViolationReport",
        "ConfidenceScore",
        "SeverityLevel",
    ]
)
