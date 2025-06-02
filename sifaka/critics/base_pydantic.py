"""PydanticAI-based base critic for Sifaka v0.3.0+

This module provides the new base critic class that uses PydanticAI agents
for structured output and seamless integration with the PydanticAI ecosystem.

Key improvements over the old BaseCritic:
- Uses PydanticAI agents for generation with structured output
- Returns CriticResult objects instead of dictionaries
- Pure async implementation
- Tool integration for real-time feedback
- No backward compatibility code
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic_ai import Agent

from sifaka.core.interfaces import Critic
from sifaka.core.thought import Thought
from sifaka.models.critic_results import (
    ConfidenceScore,
    CriticResult,
    CritiqueFeedback,
    ImprovementSuggestion,
    SeverityLevel,
    ViolationReport,
)
from sifaka.utils.error_handling import ImproverError, critic_context
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin
from sifaka.critics.mixins.validation_aware import ValidationAwareMixin

logger = get_logger(__name__)


class PydanticAICritic(ContextAwareMixin, ValidationAwareMixin, Critic, ABC):
    """Modern PydanticAI-based base critic for Sifaka v0.3.0+

    This class provides the foundation for all modern Sifaka critics using
    PydanticAI agents with structured output. It replaces the old BaseCritic
    with a clean, async-first implementation.

    Key features:
    - PydanticAI agent for structured output generation
    - Returns CriticResult objects with rich metadata
    - Pure async implementation
    - Context and validation awareness
    - Tool integration support
    - No backward compatibility
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the PydanticAI critic.

        Args:
            model_name: The model name for the PydanticAI agent (e.g., "openai:gpt-4")
            system_prompt: Optional system prompt for the agent
            **agent_kwargs: Additional arguments passed to the PydanticAI agent
        """
        super().__init__()

        self.model_name = model_name
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Create PydanticAI agent with structured output
        self.agent = Agent(
            model=model_name,
            output_type=CritiqueFeedback,
            system_prompt=self.system_prompt,
            **agent_kwargs,
        )

        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")

    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for this critic type.

        Returns:
            The default system prompt string.
        """

    @abstractmethod
    async def _create_critique_prompt(self, thought: Thought) -> str:
        """Create the critique prompt for the given thought.

        Args:
            thought: The thought to critique.

        Returns:
            The formatted critique prompt.
        """

    async def critique_async(self, thought: Thought) -> CriticResult:
        """Critique text using PydanticAI agent with structured output.

        Args:
            thought: The Thought container with the text to critique.

        Returns:
            A CriticResult with structured feedback.
        """
        start_time = time.time()

        with critic_context(
            critic_name=self.__class__.__name__,
            operation="critique_async",
            message_prefix=f"Failed to critique text with {self.__class__.__name__}",
        ):
            try:
                # Check if text is available
                if not thought.text:
                    return self._create_error_result(
                        "No text available for critique",
                        start_time=start_time,
                    )

                # Create critique prompt
                critique_prompt = await self._create_critique_prompt(thought)

                # Run PydanticAI agent with structured output
                result = await self.agent.run(critique_prompt)

                # Extract structured feedback
                feedback = result.output

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000

                # Create CriticResult
                return CriticResult(
                    feedback=feedback,
                    operation_type="critique",
                    success=True,
                    total_processing_time_ms=processing_time,
                    model_calls=1,
                    input_text_length=len(thought.text),
                    validation_context=self._get_validation_context_dict(thought),
                    metadata={
                        "model_name": self.model_name,
                        "critic_name": self.__class__.__name__,
                        "prompt_length": len(critique_prompt),
                    },
                )

            except Exception as e:
                logger.error(f"{self.__class__.__name__} critique failed: {e}")
                return self._create_error_result(
                    f"Critique failed: {str(e)}",
                    start_time=start_time,
                    error_type=type(e).__name__,
                )

    async def improve_async(self, thought: Thought) -> str:
        """Improve text based on critique using PydanticAI agent.

        Args:
            thought: The Thought container with the text to improve.

        Returns:
            The improved text.

        Raises:
            ImproverError: If the improvement fails.
        """
        try:
            # First get critique feedback
            critique_result = await self.critique_async(thought)

            if not critique_result.success:
                raise ImproverError(
                    f"Cannot improve: critique failed - {critique_result.error_message}"
                )

            # Create improvement prompt
            improvement_prompt = await self._create_improvement_prompt(
                thought, critique_result.feedback
            )

            # Create improvement agent (returns string, not structured output)
            improvement_agent = Agent(
                model=self.model_name,
                output_type=str,
                system_prompt="You are an expert text improver. Generate improved text based on critique feedback.",
            )

            # Run improvement
            result = await improvement_agent.run(improvement_prompt)

            return result.output

        except Exception as e:
            logger.error(f"{self.__class__.__name__} improvement failed: {e}")
            raise ImproverError(f"Improvement failed: {str(e)}")

    async def _create_improvement_prompt(self, thought: Thought, feedback: CritiqueFeedback) -> str:
        """Create the improvement prompt based on critique feedback.

        Args:
            thought: The original thought.
            feedback: The critique feedback.

        Returns:
            The formatted improvement prompt.
        """
        # Prepare context
        context = self._prepare_context(thought)

        # Get validation context if available
        validation_context = self._get_validation_context_dict(thought)
        validation_text = ""
        if validation_context:
            validation_text = f"\n\nValidation Context:\n{validation_context}"

        # Format violations and suggestions
        violations_text = ""
        if feedback.violations:
            violations_text = "\n\nViolations Found:\n" + "\n".join(
                f"- {v.description} (Severity: {v.severity})" for v in feedback.violations
            )

        suggestions_text = ""
        if feedback.suggestions:
            suggestions_text = "\n\nImprovement Suggestions:\n" + "\n".join(
                f"- {s.suggestion} (Priority: {s.priority})" for s in feedback.suggestions
            )

        return f"""Please improve the following text based on the critique feedback.

Original Prompt: {thought.prompt}

Current Text:
{thought.text}

Context:
{context}

Critique Feedback:
{feedback.message}
{violations_text}
{suggestions_text}
{validation_text}

Please provide an improved version of the text that addresses the critique feedback while maintaining the original intent and style."""

    def _create_error_result(
        self,
        error_message: str,
        start_time: float,
        error_type: Optional[str] = None,
    ) -> CriticResult:
        """Create an error CriticResult.

        Args:
            error_message: The error message.
            start_time: When the operation started.
            error_type: Optional error type.

        Returns:
            A CriticResult indicating failure.
        """
        processing_time = (time.time() - start_time) * 1000

        # Create minimal feedback for error case
        error_feedback = CritiqueFeedback(
            message=error_message,
            needs_improvement=True,
            confidence=ConfidenceScore(overall=0.0),
            critic_name=self.__class__.__name__,
        )

        return CriticResult(
            feedback=error_feedback,
            operation_type="critique",
            success=False,
            error_message=error_message,
            error_type=error_type,
            total_processing_time_ms=processing_time,
            model_calls=0,
            input_text_length=0,
            metadata={
                "critic_name": self.__class__.__name__,
                "error": True,
            },
        )

    def _get_validation_context_dict(self, thought: Thought) -> Optional[Dict[str, Any]]:
        """Get validation context as a dictionary.

        Args:
            thought: The thought to get validation context from.

        Returns:
            Validation context dictionary or None.
        """
        if not thought.validation_results:
            return None

        return {
            validator_name: {
                "is_valid": result.is_valid,
                "message": result.message,
                "confidence": result.confidence,
                "violations": result.violations,
            }
            for validator_name, result in thought.validation_results.items()
        }
