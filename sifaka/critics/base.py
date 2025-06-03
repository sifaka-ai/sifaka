"""Base critic implementation for Sifaka.

This module provides the base critic class that all Sifaka critics inherit from.
It provides common functionality including PydanticAI integration, structured output,
optional retrieval augmentation, and consistent error handling.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from sifaka.core.interfaces import Critic
from sifaka.core.thought import SifakaThought, ValidationContext
from sifaka.utils.validation_aware import ValidationAwareMixin


class CritiqueFeedback(BaseModel):
    """Structured feedback from a critic.

    This model defines the standard output format for all critics,
    ensuring consistent and structured feedback across different
    critique methodologies.
    """

    needs_improvement: bool
    message: str
    suggestions: List[str]
    confidence: float  # 0.0 to 1.0
    reasoning: str
    # Note: metadata removed from structured output to avoid Gemini additionalProperties issue
    # Metadata is now handled internally by the critic after receiving the structured response

    @property
    def metadata(self) -> Dict[str, Any]:
        """Compatibility property for metadata access.

        Returns empty dict since metadata is handled separately in the critic system.
        This property exists for backward compatibility with code that expects
        a metadata attribute on CritiqueFeedback objects.
        """
        return {}


class BaseCritic(ValidationAwareMixin, Critic, ABC):
    """Base class for all Sifaka critics.

    This class provides the foundation for implementing research-based critics
    using PydanticAI agents. It handles:

    - PydanticAI agent creation and management
    - Structured output using CritiqueFeedback model
    - Optional retrieval augmentation
    - Validation-aware prompting and suggestion filtering
    - Consistent error handling
    - Standard critique interface

    All critics should inherit from this class and implement the abstract methods.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        paper_reference: str,
        methodology: str,
        retrieval_tools: Optional[List[Any]] = None,
        **agent_kwargs: Any,
    ):
        """Initialize the base critic.

        Args:
            model_name: The model name for the PydanticAI agent (e.g., "openai:gpt-4o-mini")
            system_prompt: The system prompt that defines the critic's behavior
            paper_reference: Full citation of the research paper this critic implements
            methodology: Description of the methodology being implemented
            retrieval_tools: Optional list of retrieval tools for RAG support
            **agent_kwargs: Additional arguments passed to the PydanticAI agent
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.paper_reference = paper_reference
        self.methodology = methodology
        self.retrieval_tools = retrieval_tools or []

        # Create PydanticAI agent with structured output
        self.agent = Agent(
            model=model_name,
            output_type=CritiqueFeedback,
            system_prompt=system_prompt,
            tools=self.retrieval_tools,
            **agent_kwargs,
        )

        # Store metadata about this critic
        self.metadata = {
            "critic_type": self.__class__.__name__,
            "model_name": model_name,
            "has_retrieval": len(self.retrieval_tools) > 0,
            "paper_reference": paper_reference,
            "methodology": methodology,
        }

    async def critique_async(self, thought: SifakaThought) -> None:
        """Critique the current text in a thought and add results to the thought.

        This is the main interface method that all critics must implement.
        It analyzes the thought's current text and adds structured feedback
        directly to the thought's audit trail.

        Args:
            thought: The SifakaThought containing text to critique
        """
        import time

        start_time = time.time()
        tools_used = []
        retrieval_context = None

        try:
            # Build the critique prompt
            prompt = await self._build_critique_prompt(thought)

            # Run the PydanticAI agent
            result = await self.agent.run(prompt)

            # Extract feedback
            feedback = result.output

            # Track tool usage if any
            if hasattr(result, "tool_calls") and result.tool_calls:
                tools_used = [call.tool_name for call in result.tool_calls]
                # Extract retrieval context from tool results
                retrieval_context = self._extract_retrieval_context(result.tool_calls)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Apply validation-aware filtering to suggestions
            filtered_feedback = self._apply_validation_filtering(thought, feedback)

            # Add critique to thought with rich metadata
            thought.add_critique(
                critic=self.__class__.__name__,
                feedback=filtered_feedback.message,
                suggestions=filtered_feedback.suggestions,
                confidence=filtered_feedback.confidence,
                reasoning=filtered_feedback.reasoning,
                needs_improvement=filtered_feedback.needs_improvement,
                critic_metadata=self._get_critic_specific_metadata(filtered_feedback),
                processing_time_ms=processing_time_ms,
                model_name=self.model_name,
                paper_reference=self.paper_reference,
                methodology=self.methodology,
                tools_used=tools_used,
                retrieval_context=retrieval_context,
            )

        except Exception as e:
            # Add error critique on failure
            processing_time_ms = (time.time() - start_time) * 1000

            thought.add_critique(
                critic=self.__class__.__name__,
                feedback=f"Critique failed: {str(e)}",
                suggestions=[],
                confidence=0.0,
                reasoning=f"Error occurred during critique: {str(e)}",
                needs_improvement=False,
                critic_metadata={"error": str(e), "error_type": type(e).__name__},
                processing_time_ms=processing_time_ms,
                model_name=self.model_name,
                paper_reference=self.paper_reference,
                methodology=self.methodology,
                tools_used=tools_used,
                retrieval_context=retrieval_context,
            )

    @abstractmethod
    async def _build_critique_prompt(self, thought: SifakaThought) -> str:
        """Build the critique prompt for this specific critic.

        Each critic implementation must define how to construct the prompt
        that will be sent to the PydanticAI agent. This should include:

        - The text to be critiqued
        - Any relevant context from the thought
        - Specific instructions for this critic type
        - Validation context if available

        Args:
            thought: The SifakaThought containing text and context

        Returns:
            The formatted prompt string
        """
        pass

    def _get_critic_specific_metadata(self, feedback: CritiqueFeedback) -> Dict[str, Any]:
        """Extract critic-specific metadata from the feedback.

        Subclasses can override this to add specific metadata fields.

        Args:
            feedback: The structured feedback from the critic

        Returns:
            Dictionary of critic-specific metadata
        """
        return {
            "feedback_length": len(feedback.message),
            "num_suggestions": len(feedback.suggestions),
            "confidence_level": (
                "high"
                if feedback.confidence > 0.8
                else "medium" if feedback.confidence > 0.5 else "low"
            ),
        }

    def _extract_retrieval_context(self, tool_calls: List[Any]) -> Optional[Dict[str, Any]]:
        """Extract retrieval context from tool calls.

        Args:
            tool_calls: List of tool calls made during critique

        Returns:
            Dictionary containing retrieval context or None
        """
        if not tool_calls:
            return None

        context = {
            "num_retrievals": len(tool_calls),
            "tools_called": [call.tool_name for call in tool_calls],
            "retrieval_results": [],
        }

        for call in tool_calls:
            if hasattr(call, "result"):
                context["retrieval_results"].append(
                    {
                        "tool": call.tool_name,
                        "result_summary": (
                            str(call.result)[:200] + "..."
                            if len(str(call.result)) > 200
                            else str(call.result)
                        ),
                    }
                )

        return context

    def _get_validation_context(self, thought: SifakaThought) -> str:
        """Extract validation context from the thought for prompt inclusion.

        Args:
            thought: The SifakaThought to extract validation context from

        Returns:
            Formatted validation context string with priority awareness
        """
        # Use the new validation-aware context method from mixin
        return self._get_validation_aware_context(thought)

    def _get_previous_critiques(self, thought: SifakaThought) -> str:
        """Extract previous critique context from the thought.

        Args:
            thought: The SifakaThought to extract critique history from

        Returns:
            Formatted previous critiques string
        """
        if thought.iteration == 0:
            return ""

        prev_iteration = thought.iteration - 1
        prev_critiques = [c for c in thought.critiques if c.iteration == prev_iteration]

        if not prev_critiques:
            return ""

        context_parts = ["PREVIOUS CRITIQUE FEEDBACK:"]
        for critique in prev_critiques[-3:]:  # Limit to last 3 critiques
            context_parts.append(f"- {critique.critic}: {critique.feedback}")
            if critique.suggestions:
                for suggestion in critique.suggestions[:2]:  # Limit suggestions
                    context_parts.append(f"  → {suggestion}")

        return "\n".join(context_parts)

    def add_retrieval_tool(self, tool: Any) -> None:
        """Add a retrieval tool to this critic for RAG support.

        Args:
            tool: A PydanticAI tool for retrieval augmentation
        """
        self.retrieval_tools.append(tool)
        # Recreate agent with new tools
        self.agent = Agent(
            model=self.model_name,
            output_type=CritiqueFeedback,
            system_prompt=self.system_prompt,
            tools=self.retrieval_tools,
        )
        self.metadata["has_retrieval"] = True

    def _apply_validation_filtering(
        self, thought: SifakaThought, feedback: CritiqueFeedback
    ) -> CritiqueFeedback:
        """Apply validation-aware filtering to critique feedback.

        Args:
            thought: The SifakaThought containing validation context
            feedback: The original critique feedback

        Returns:
            Filtered critique feedback with validation-aware suggestions
        """
        # Extract validation context
        validation_context = ValidationContext.extract_constraints(thought)

        if not validation_context:
            # No validation constraints, return original feedback
            return feedback

        # Filter suggestions that conflict with validation constraints
        filtered_suggestions = self._filter_suggestions_for_constraints(
            feedback.suggestions, validation_context
        )

        # Update feedback message if suggestions were filtered
        updated_message = feedback.message
        if len(filtered_suggestions) < len(feedback.suggestions):
            removed_count = len(feedback.suggestions) - len(filtered_suggestions)
            updated_message += (
                f"\n\nNote: {removed_count} suggestion(s) filtered due to validation constraints."
            )

        # Create new feedback with filtered suggestions
        return CritiqueFeedback(
            needs_improvement=feedback.needs_improvement,
            message=updated_message,
            suggestions=filtered_suggestions,
            confidence=feedback.confidence,
            reasoning=feedback.reasoning,
            metadata=feedback.metadata,
        )

    def get_info(self) -> Dict[str, Any]:
        """Get information about this critic.

        Returns:
            Dictionary containing critic metadata and configuration
        """
        return {
            "name": self.__class__.__name__,
            "model": self.model_name,
            "has_retrieval": len(self.retrieval_tools) > 0,
            "num_tools": len(self.retrieval_tools),
            "paper_reference": self.paper_reference,
            "methodology": self.methodology,
            "metadata": self.metadata,
        }
