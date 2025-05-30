"""PydanticAI tool wrappers for Sifaka validators and critics.

This module implements Phase 4 of the PydanticAI migration by providing
tool-based validation and criticism for maximum integration with PydanticAI.

Tool-based validation allows validators and critics to be used as PydanticAI
tools during generation, enabling real-time guidance and self-correcting
generation loops.

Example:
    ```python
    from pydantic_ai import Agent
    from sifaka.agents.tools import create_validation_tool, create_criticism_tool
    from sifaka.validators import LengthValidator
    from sifaka.critics import ReflexionCritic

    # Create tools from Sifaka components
    length_tool = create_validation_tool(LengthValidator(min_length=100))
    critic_tool = create_criticism_tool(ReflexionCritic(model=create_model("openai:gpt-4")))

    # Create agent with tools
    agent = Agent("openai:gpt-4", tools=[length_tool, critic_tool])

    # Agent can now use validation and criticism tools during generation
    result = agent.run("Write a story and validate its length")
    ```
"""

from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Critic, Validator
from sifaka.core.thought import CriticFeedback, Thought
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# PydanticAI is a required dependency
from pydantic_ai import Agent
from pydantic_ai.tools import Tool


def create_validation_tool(validator: Validator, tool_name: Optional[str] = None) -> "Tool":
    """Create a PydanticAI tool from a Sifaka validator.

    This function wraps a Sifaka validator as a PydanticAI tool, enabling
    real-time validation during generation.

    Args:
        validator: The Sifaka validator to wrap as a tool.
        tool_name: Optional custom tool name. Defaults to validator class name.

    Returns:
        A PydanticAI Tool that can be used by agents.

    """

    name = tool_name or f"validate_{validator.__class__.__name__.lower()}"

    async def validation_tool(text: str) -> Dict[str, Any]:
        """Validate text using the wrapped Sifaka validator.

        Args:
            text: The text to validate.

        Returns:
            Dictionary containing validation results.
        """
        logger.debug(f"Running validation tool: {name}")

        # Create temporary thought for validation
        thought = Thought(prompt="", text=text, chain_id="tool-validation")

        try:
            # Use async validation if available, otherwise fall back to sync
            if hasattr(validator, "_validate_async"):
                result = await validator._validate_async(thought)  # type: ignore
            else:
                result = validator.validate(thought)

            # Convert ValidationResult to tool-friendly format
            tool_result = {
                "validator_name": result.validator_name,
                "passed": result.passed,
                "issues": result.issues,
                "suggestions": result.suggestions,
                "confidence": result.confidence,
                "metadata": result.metadata,
            }

            logger.debug(f"Validation tool result: {'PASSED' if result.passed else 'FAILED'}")
            return tool_result

        except Exception as e:
            logger.error(f"Validation tool failed: {e}")
            return {
                "validator_name": validator.__class__.__name__,
                "passed": False,
                "issues": [str(e)],
                "suggestions": ["Please check the validator configuration"],
                "confidence": 0.0,
                "metadata": {"error": True},
            }

    # Create PydanticAI tool
    tool = Tool(
        name=name,
        description=f"Validate text using {validator.__class__.__name__}",
        function=validation_tool,
    )

    logger.debug(f"Created validation tool: {name}")
    return tool


def create_criticism_tool(critic: Critic, tool_name: Optional[str] = None) -> "Tool":
    """Create a PydanticAI tool from a Sifaka critic.

    This function wraps a Sifaka critic as a PydanticAI tool, enabling
    real-time criticism during generation.

    Args:
        critic: The Sifaka critic to wrap as a tool.
        tool_name: Optional custom tool name. Defaults to critic class name.

    Returns:
        A PydanticAI Tool that can be used by agents.

    """

    name = tool_name or f"critique_{critic.__class__.__name__.lower()}"

    async def criticism_tool(text: str) -> Dict[str, Any]:
        """Critique text using the wrapped Sifaka critic.

        Args:
            text: The text to critique.

        Returns:
            Dictionary containing criticism feedback.
        """
        logger.debug(f"Running criticism tool: {name}")

        # Create temporary thought for criticism
        thought = Thought(prompt="", text=text, chain_id="tool-criticism")

        try:
            # Use async criticism if available, otherwise fall back to sync
            if hasattr(critic, "_critique_async"):
                result = await critic._critique_async(thought)  # type: ignore
            else:
                result = critic.critique(thought)

            # Convert result to CriticFeedback if needed
            if isinstance(result, dict):
                feedback = CriticFeedback(
                    critic_name=critic.__class__.__name__,
                    feedback=result.get("feedback", ""),
                    confidence=result.get("confidence", 0.0),
                    issues=result.get("issues", []),
                    suggestions=result.get("suggestions", []),
                    needs_improvement=result.get("needs_improvement", False),
                )
            else:
                feedback = result

            # Convert CriticFeedback to tool-friendly format
            tool_result = {
                "critic_name": feedback.critic_name,
                "feedback": feedback.feedback,
                "confidence": feedback.confidence,
                "issues": feedback.issues,
                "suggestions": feedback.suggestions,
                "needs_improvement": feedback.needs_improvement,
                "metadata": feedback.metadata,
            }

            logger.debug(f"Criticism tool result: confidence={feedback.confidence}")
            return tool_result

        except Exception as e:
            logger.error(f"Criticism tool failed: {e}")
            return {
                "critic_name": critic.__class__.__name__,
                "feedback": "Criticism failed due to an error",
                "confidence": 0.0,
                "issues": [str(e)],
                "suggestions": ["Please check the critic configuration"],
                "needs_improvement": False,
                "metadata": {"error": True},
            }

    # Create PydanticAI tool
    tool = Tool(
        name=name,
        description=f"Critique text using {critic.__class__.__name__}",
        function=criticism_tool,
    )

    logger.debug(f"Created criticism tool: {name}")
    return tool


def create_self_correcting_agent(
    base_agent: "Agent",
    validators: Optional[List[Validator]] = None,
    critics: Optional[List[Critic]] = None,
    max_correction_iterations: int = 3,
) -> "Agent":
    """Create a self-correcting agent with validation and criticism tools.

    This function creates an enhanced agent that can validate and critique
    its own output using tools, enabling self-correcting generation loops.

    Args:
        base_agent: The base PydanticAI agent to enhance.
        validators: Optional list of Sifaka validators to add as tools.
        critics: Optional list of Sifaka critics to add as tools.
        max_correction_iterations: Maximum number of self-correction iterations.

    Returns:
        An enhanced PydanticAI Agent with self-correction capabilities.

    """

    # Create tools from validators and critics
    tools = []

    if validators:
        for validator in validators:
            tool = create_validation_tool(validator)
            tools.append(tool)

    if critics:
        for critic in critics:
            tool = create_criticism_tool(critic)
            tools.append(tool)

    # Add tools to the base agent
    enhanced_agent = Agent(
        model=base_agent.model,
        system_prompt=base_agent.system_prompt,
        tools=tools,
        deps_type=base_agent.deps_type,
    )

    logger.info(f"Created self-correcting agent with {len(tools)} validation/criticism tools")
    return enhanced_agent
