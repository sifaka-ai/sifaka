"""PydanticAI model adapter for Sifaka.

This module provides an adapter to use PydanticAI agents as Sifaka models,
enabling seamless integration between the two frameworks.

Example:
    ```python
    from pydantic_ai import Agent
    from sifaka.models.pydantic_ai import PydanticAIModel

    # Create a PydanticAI agent
    agent = Agent('openai:gpt-4o', system_prompt='You are a helpful assistant.')

    # Wrap it as a Sifaka model
    model = PydanticAIModel(agent)

    # Use it like any other Sifaka model
    response = model.generate("Write a story about AI")
    print(response)
    ```
"""

from typing import Any, Optional

from sifaka.core.thought import Thought
from sifaka.utils.error_handling import ConfigurationError
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin

logger = get_logger(__name__)

# Check if PydanticAI is available
try:
    from pydantic_ai import Agent

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None


class PydanticAIModel(ContextAwareMixin):
    """Adapter to use PydanticAI agents as Sifaka models.

    This class wraps a PydanticAI agent and implements the Sifaka Model protocol,
    allowing PydanticAI agents to be used seamlessly within Sifaka chains.

    The adapter handles:
    - Converting Sifaka prompts to PydanticAI format
    - Extracting text output from PydanticAI results
    - Token counting approximation
    - Context integration from Thought objects
    """

    def __init__(self, agent: "Agent", model_name: Optional[str] = None):
        """Initialize the PydanticAI model adapter.

        Args:
            agent: The PydanticAI agent to wrap.
            model_name: Optional custom model name. If not provided, will be
                       derived from the agent's model.

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

        super().__init__()
        self.agent = agent
        self.model_name = model_name or self._derive_model_name(agent)

    def _derive_model_name(self, agent: "Agent") -> str:
        """Derive a model name from the PydanticAI agent."""
        try:
            # Try to get the model name from the agent
            if hasattr(agent, "model") and agent.model:
                return f"pydantic-ai-{agent.model}"
            else:
                return "pydantic-ai-agent"
        except Exception:
            return "pydantic-ai-agent"

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text using PydanticAI agent.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options passed to the agent.

        Returns:
            The generated text.
        """
        logger.debug(f"Generating text with PydanticAI agent: {self.model_name}")

        try:
            # Run the agent synchronously
            result = self.agent.run_sync(prompt, **options)

            # Extract the output text
            output = self._extract_output(result)

            logger.debug(f"Generated text: {len(output)} characters")
            return output

        except Exception as e:
            logger.error(f"PydanticAI generation failed: {e}")
            raise

    def generate_with_rich_result(self, prompt: str, **options: Any) -> tuple[str, dict]:
        """Generate text using PydanticAI agent and return rich result data.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options passed to the agent.

        Returns:
            A tuple of (generated_text, rich_result_data).
            The rich_result_data contains usage, cost, messages, tool_calls, and metadata.
        """
        logger.debug(f"Generating text with rich result capture: {self.model_name}")

        try:
            # Run the agent synchronously
            result = self.agent.run_sync(prompt, **options)

            # Extract rich result data
            rich_data = self._extract_rich_result(result)

            logger.debug(f"Generated text with rich data: {len(rich_data['output'])} characters")
            return rich_data["output"], rich_data

        except Exception as e:
            logger.error(f"PydanticAI rich generation failed: {e}")
            raise

    def generate_with_thought(self, thought: Thought, **options: Any) -> tuple[str, str]:
        """Generate text using Thought context.

        This method converts the Thought context into a format suitable for
        PydanticAI and includes relevant context in the prompt.

        Args:
            thought: The Thought container with context for generation.
            **options: Additional options passed to the agent.

        Returns:
            A tuple of (generated_text, actual_prompt_used).
        """
        logger.debug("Generating text with thought context")

        # Build enhanced prompt from thought
        enhanced_prompt = self._build_prompt_from_thought(thought)

        try:
            # Run the agent with enhanced prompt
            result = self.agent.run_sync(enhanced_prompt, **options)

            # Extract the output text
            output = self._extract_output(result)

            logger.debug(f"Generated text with thought: {len(output)} characters")
            return output, enhanced_prompt

        except Exception as e:
            logger.error(f"PydanticAI generation with thought failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Estimate token count for the given text.

        This provides a rough approximation since PydanticAI doesn't expose
        token counting directly. For more accurate counts, use the underlying
        model's tokenizer directly.

        Args:
            text: The text to count tokens in.

        Returns:
            Estimated number of tokens.
        """
        # Simple approximation: words * 1.3 (rough token-to-word ratio)
        word_count = len(text.split())
        estimated_tokens = int(word_count * 1.3)

        logger.debug(f"Estimated tokens: {estimated_tokens} for {word_count} words")
        return estimated_tokens

    # Async methods required by critics
    async def _generate_async(self, prompt: str, **options: Any) -> str:
        """Generate text asynchronously (required by critics).

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options passed to the agent.

        Returns:
            The generated text.
        """
        logger.debug(f"Generating text asynchronously with PydanticAI agent: {self.model_name}")

        try:
            # Run the agent asynchronously
            result = await self.agent.run(prompt, **options)

            # Extract the output text
            output = self._extract_output(result)

            logger.debug(f"Generated text async: {len(output)} characters")
            return output

        except Exception as e:
            logger.error(f"PydanticAI async generation failed: {e}")
            raise

    async def generate_with_rich_result_async(
        self, prompt: str, **options: Any
    ) -> tuple[str, dict]:
        """Generate text asynchronously with rich result data.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options passed to the agent.

        Returns:
            A tuple of (generated_text, rich_result_data).
            The rich_result_data contains usage, cost, messages, tool_calls, and metadata.
        """
        logger.debug(f"Generating text async with rich result capture: {self.model_name}")

        try:
            # Run the agent asynchronously
            result = await self.agent.run(prompt, **options)

            # Extract rich result data
            rich_data = self._extract_rich_result(result)

            logger.debug(
                f"Generated text async with rich data: {len(rich_data['output'])} characters"
            )
            return rich_data["output"], rich_data

        except Exception as e:
            logger.error(f"PydanticAI async rich generation failed: {e}")
            raise

    async def _generate_with_thought_async(
        self, thought: Thought, **options: Any
    ) -> tuple[str, str]:
        """Generate text from a thought asynchronously.

        Args:
            thought: The Thought container with context for generation.
            **options: Additional options passed to the agent.

        Returns:
            A tuple of (generated_text, actual_prompt_used).
        """
        logger.debug("Generating text asynchronously with thought context")

        # Build enhanced prompt from thought
        enhanced_prompt = self._build_prompt_from_thought(thought)

        try:
            # Run the agent with enhanced prompt
            result = await self.agent.run(enhanced_prompt, **options)

            # Extract the output text
            output = self._extract_output(result)

            logger.debug(f"Generated text async with thought: {len(output)} characters")
            return output, enhanced_prompt

        except Exception as e:
            logger.error(f"PydanticAI async generation with thought failed: {e}")
            raise

    async def _count_tokens_async(self, text: str) -> int:
        """Count tokens asynchronously (for consistency with other models).

        Args:
            text: The text to count tokens in.

        Returns:
            Estimated number of tokens.
        """
        return self.count_tokens(text)

    def _build_prompt_from_thought(self, thought: Thought) -> str:
        """Build an enhanced prompt from a Thought object.

        This method incorporates context from the thought including:
        - Original prompt
        - Retrieved documents (if any)
        - Previous iterations context
        - Critic feedback (if available)

        Args:
            thought: The thought object containing context.

        Returns:
            Enhanced prompt string.
        """
        prompt_parts = []

        # Start with the original prompt
        if thought.prompt:
            prompt_parts.append(thought.prompt)

        # Add retrieved context if available
        if hasattr(thought, "retrieved_documents") and thought.retrieved_documents:
            prompt_parts.append("\nRelevant context:")
            for doc in thought.retrieved_documents[:3]:  # Limit to top 3 docs
                prompt_parts.append(f"- {doc}")

        # Add previous iteration context if this is an improvement
        if thought.iteration > 1 and thought.text:
            prompt_parts.append(f"\nPrevious attempt:\n{thought.text}")

            # Add critic feedback if available
            if hasattr(thought, "critic_feedback") and thought.critic_feedback:
                prompt_parts.append("\nFeedback to address:")
                for feedback in thought.critic_feedback[-2:]:  # Last 2 feedbacks
                    if hasattr(feedback, "feedback") and feedback.feedback:
                        prompt_parts.append(f"- {feedback.feedback}")

        return "\n".join(prompt_parts)

    def _extract_output(self, result) -> str:
        """Extract text output from PydanticAI result.

        Args:
            result: The result object from PydanticAI agent.run_sync()

        Returns:
            The extracted text output.
        """
        try:
            # PydanticAI results have an 'output' attribute
            if hasattr(result, "output"):
                output = result.output

                # Handle different output types
                if isinstance(output, str):
                    return output
                elif hasattr(output, "__str__"):
                    return str(output)
                else:
                    # For structured outputs, convert to string representation
                    return repr(output)
            else:
                # Fallback: convert entire result to string
                return str(result)

        except Exception as e:
            logger.warning(f"Failed to extract output from PydanticAI result: {e}")
            return str(result)

    def _extract_rich_result(self, result) -> dict:
        """Extract rich metadata from PydanticAI AgentRunResult.

        Args:
            result: The AgentRunResult object from PydanticAI agent.run_sync()

        Returns:
            Dictionary containing rich result data including usage, cost, messages, etc.
        """
        rich_data = {
            "output": self._extract_output(result),
            "usage": None,
            "cost": None,
            "messages": None,
            "tool_calls": None,
            "metadata": {},
        }

        try:
            # Extract usage information (token counts)
            if hasattr(result, "usage") and result.usage:
                rich_data["usage"] = {
                    "requests": getattr(result.usage, "requests", None),
                    "request_tokens": getattr(result.usage, "request_tokens", None),
                    "response_tokens": getattr(result.usage, "response_tokens", None),
                    "total_tokens": getattr(result.usage, "total_tokens", None),
                }

            # Extract cost information
            if hasattr(result, "cost") and result.cost:
                rich_data["cost"] = {
                    "request_cost": getattr(result.cost, "request_cost", None),
                    "response_cost": getattr(result.cost, "response_cost", None),
                    "total_cost": getattr(result.cost, "total_cost", None),
                    "details": getattr(result.cost, "details", None),
                }

            # Extract message history
            if hasattr(result, "messages") and result.messages:
                rich_data["messages"] = []
                for msg in result.messages:
                    msg_data = {
                        "role": getattr(msg, "role", None),
                        "content": getattr(msg, "content", None),
                        "timestamp": getattr(msg, "timestamp", None),
                    }
                    # Add tool call information if present
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        msg_data["tool_calls"] = [
                            {
                                "name": getattr(tc, "name", None),
                                "args": getattr(tc, "args", None),
                                "result": getattr(tc, "result", None),
                            }
                            for tc in msg.tool_calls
                        ]
                    rich_data["messages"].append(msg_data)

            # Extract any additional metadata
            for attr in ["model", "timestamp", "run_id"]:
                if hasattr(result, attr):
                    rich_data["metadata"][attr] = getattr(result, attr)

        except Exception as e:
            logger.warning(f"Failed to extract rich data from PydanticAI result: {e}")
            # Ensure we always have the basic output even if rich extraction fails
            rich_data["output"] = self._extract_output(result)

        return rich_data


def create_pydantic_ai_model(agent: "Agent", **kwargs) -> PydanticAIModel:
    """Factory function to create a PydanticAI model adapter.

    Args:
        agent: The PydanticAI agent to wrap.
        **kwargs: Additional arguments passed to PydanticAIModel.

    Returns:
        A PydanticAIModel instance.
    """
    return PydanticAIModel(agent=agent, **kwargs)
