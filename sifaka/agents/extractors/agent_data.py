"""Agent data extraction utilities.

This module handles extracting data from PydanticAI agent results,
including output text, metadata, and tool calls.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from pydantic_ai import Agent

from sifaka.core.thought import Thought, ToolCall
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class AgentDataExtractor:
    """Extracts data from PydanticAI agent results."""

    def __init__(self, agent: Agent):
        """Initialize the data extractor.

        Args:
            agent: The PydanticAI agent to extract data from.
        """
        self.agent = agent

    def extract_output(
        self, result: Any, thought: Optional[Thought] = None
    ) -> Tuple[str, Optional[Thought]]:
        """Extract text output from PydanticAI agent result and log tool calls.

        Args:
            result: The agent result object.
            thought: Optional thought to add tool calls to.

        Returns:
            Tuple of (extracted_text_output, updated_thought_with_tool_calls).
        """
        updated_thought = thought

        try:
            # Extract tool calls if available
            if hasattr(result, "all_messages") and updated_thought is not None:
                updated_thought = self._extract_tool_calls(result, updated_thought)

            # Extract output text
            if hasattr(result, "output"):
                output = result.output
                if isinstance(output, str):
                    return output, updated_thought
                else:
                    return str(output), updated_thought
            else:
                return str(result), updated_thought

        except Exception as e:
            logger.warning(f"Failed to extract agent output: {e}")
            return str(result), updated_thought

    def extract_metadata(self, result: Any, fallback_prompt: str) -> Dict[str, str]:
        """Extract metadata from PydanticAI agent result.

        Args:
            result: The agent result object.
            fallback_prompt: Fallback prompt if extraction fails.

        Returns:
            Dictionary containing model_name, model_prompt, and system_prompt.
        """
        return {
            "model_name": self._extract_model_name(),
            "model_prompt": self._extract_model_prompt(result, fallback_prompt),
            "system_prompt": self._extract_system_prompt(result),
        }

    def _extract_model_name(self) -> str:
        """Extract model name from PydanticAI agent.

        Returns:
            The model name string.
        """
        try:
            # Try to get model name from the agent
            if hasattr(self.agent, "model"):
                model = self.agent.model
                if hasattr(model, "_model_name"):
                    return f"pydantic-ai:{model._model_name}"
                elif hasattr(model, "name"):
                    return f"pydantic-ai:{model.name}"
                else:
                    return f"pydantic-ai:{str(model)}"
            else:
                return "pydantic-ai-agent"
        except Exception as e:
            logger.warning(f"Failed to extract model name: {e}")
            return "pydantic-ai-agent"

    def _extract_model_prompt(self, result: Any, fallback_prompt: str) -> str:
        """Extract the actual prompt sent to the model from PydanticAI result.

        Args:
            result: The agent result object.
            fallback_prompt: Fallback prompt if extraction fails.

        Returns:
            The model prompt string.
        """
        try:
            # Try to get the actual prompt from messages
            if hasattr(result, "all_messages"):
                messages = result.all_messages()
                for message in messages:
                    if hasattr(message, "content") and hasattr(message, "role"):
                        if message.role == "user":
                            return str(message.content)

            # Fallback to the original prompt
            return fallback_prompt

        except Exception as e:
            logger.warning(f"Failed to extract model prompt: {e}")
            return fallback_prompt

    def _extract_system_prompt(self, result: Any) -> str:
        """Extract system prompt from PydanticAI result.

        Args:
            result: The agent result object.

        Returns:
            The system prompt string.
        """
        try:
            # Try to get system prompt from agent's internal storage first
            if hasattr(self.agent, "_system_prompts") and self.agent._system_prompts:
                # PydanticAI stores system prompts in a tuple
                return str(self.agent._system_prompts[0])

            # Fallback: Try to get system prompt from agent
            if hasattr(self.agent, "system_prompt"):
                system_prompt = self.agent.system_prompt
                if callable(system_prompt):
                    # For callable system prompts, try to call it or get its string representation
                    try:
                        # Try calling with no arguments first
                        return str(system_prompt())
                    except Exception:
                        try:
                            # Try to get the actual prompt text from the agent's internal state
                            if hasattr(self.agent, "_system_prompt"):
                                return str(self.agent._system_prompt)
                            # Try to extract from the function if it's a simple string wrapper
                            elif (
                                hasattr(system_prompt, "__closure__") and system_prompt.__closure__
                            ):
                                for cell in system_prompt.__closure__:
                                    if isinstance(cell.cell_contents, str):
                                        return cell.cell_contents
                            # Fallback to a descriptive string
                            return "Dynamic system prompt (callable)"
                        except Exception:
                            return "Dynamic system prompt (callable)"
                else:
                    return str(system_prompt) if system_prompt else ""

            # Try to get from messages
            if hasattr(result, "all_messages"):
                messages = result.all_messages()
                for message in messages:
                    if hasattr(message, "content") and hasattr(message, "role"):
                        if message.role == "system":
                            return str(message.content)

            return ""

        except Exception as e:
            logger.warning(f"Failed to extract system prompt: {e}")
            return ""

    def _extract_tool_calls(self, result: Any, thought: Thought) -> Thought:
        """Extract tool calls from PydanticAI result and add to thought.

        Args:
            result: The agent result object.
            thought: The thought to add tool calls to.

        Returns:
            Updated thought with tool calls.
        """
        try:
            if hasattr(result, "all_messages"):
                for message in result.all_messages():
                    # Check for tool calls in the message
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            # Extract tool call information based on PydanticAI structure
                            tool_name = "unknown"
                            arguments = {}

                            if hasattr(tool_call, "function"):
                                func = tool_call.function
                                if hasattr(func, "name"):
                                    tool_name = func.name
                                if hasattr(func, "arguments"):
                                    arguments = func.arguments
                            elif hasattr(tool_call, "name"):
                                tool_name = tool_call.name
                                if hasattr(tool_call, "arguments"):
                                    arguments = tool_call.arguments

                            tool_record = ToolCall(
                                tool_name=tool_name,
                                arguments=arguments,
                                timestamp=datetime.now(),
                                success=True,  # Assume success if we got here
                            )
                            thought = thought.add_tool_call(tool_record)

            return thought

        except Exception as e:
            logger.warning(f"Failed to extract tool calls: {e}")
            return thought
