"""Text generation and improvement component of the Sifaka engine.

This module handles the generation of improved text based on critic feedback
and validation results. It's the component that actually creates new versions
of text during the improvement process.

## Architecture:

The generator works in conjunction with critics and validators:
1. Critics provide feedback on what needs improvement
2. Validators identify specific requirements not met
3. Generator synthesizes all feedback into an improved version

## Key Components:

- **TextGenerator**: Main class that handles text improvement
- **ImprovementResponse**: Structured response with confidence tracking
- **Prompt Building**: Sophisticated prompt construction from feedback

## Design Decisions:

1. **Structured Output**: Uses PydanticAI for type-safe improvements
2. **Feedback Prioritization**: Recent feedback weighted more heavily
3. **Deduplication**: Avoids repeating feedback from same critic
4. **Metadata Integration**: Incorporates critic-specific insights

## Usage:

    >>> generator = TextGenerator(model="gpt-4", temperature=0.7)
    >>> improved_text, prompt, tokens, time = await generator.generate_improvement(
    ...     current_text="Original text",
    ...     result=sifaka_result  # Contains all feedback
    ... )

## Performance Considerations:

- Caches LLM client for reuse across iterations
- Limits feedback context to prevent prompt bloat
- Tracks token usage for cost monitoring
"""

import time
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from ..exceptions import ModelProviderError
from ..llm_client import LLMClient, LLMManager
from ..models import CritiqueResult, SifakaResult

# Import logfire if available
try:
    import logfire
except ImportError:
    logfire = None  # type: ignore[assignment]


class ImprovementResponse(BaseModel):
    """Structured response model for text improvements.

    This model ensures that improvements are returned in a consistent
    format with metadata about what changes were made. Using structured
    output improves reliability and enables better tracking.

    Example:
        >>> response = ImprovementResponse(
        ...     improved_text="The enhanced text with improvements",
        ...     changes_made=[
        ...         "Added clarity to introduction",
        ...         "Fixed grammar issues",
        ...         "Improved sentence flow"
        ...     ],
        ...     confidence=0.85
        ... )

    Attributes:
        improved_text: The new version of text after improvements
        changes_made: List of specific changes applied (for transparency)
        confidence: Generator's confidence in the improvements (0.0-1.0)
    """

    improved_text: str = Field(..., description="The improved version of the text")
    changes_made: list[str] = Field(
        default_factory=list, description="List of changes made"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in improvements"
    )


class TextGenerator:
    """Core component for generating improved text based on feedback.

    The TextGenerator is responsible for taking the current text and all
    accumulated feedback (from critics and validators) and producing an
    improved version. It uses sophisticated prompt engineering to ensure
    all feedback is addressed while maintaining text coherence.

    Key features:
    - Integrates feedback from multiple critics
    - Prioritizes validation failures
    - Deduplicates redundant feedback
    - Tracks improvement confidence
    - Handles critic-specific metadata

    Example:
        >>> # Initialize generator
        >>> generator = TextGenerator(
        ...     model="gpt-4",
        ...     temperature=0.7  # Balance creativity and consistency
        ... )
        >>>
        >>> # Generate improvement
        >>> text, prompt, tokens, time = await generator.generate_improvement(
        ...     current_text="Current version",
        ...     result=result_with_feedback
        ... )
        >>>
        >>> if text:
        ...     print(f"Improved in {time:.2f}s using {tokens} tokens")

    The generator uses a carefully crafted system prompt that emphasizes
    iterative improvement and attention to all feedback sources.
    """

    IMPROVEMENT_SYSTEM_PROMPT = """You are an expert text editor focused on iterative improvement. Pay careful attention to all critic feedback and validation issues. Your goal is to address each piece of feedback thoroughly while maintaining the original intent and improving the overall quality of the text."""

    def __init__(self, model: str, temperature: float, provider: Optional[str] = None):
        """Initialize the text generator with model configuration.

        Args:
            model: Name of the LLM model to use for generation.
                Examples: "gpt-4", "claude-3-opus", "gpt-3.5-turbo".
                The model should be capable of following complex
                instructions and producing high-quality text.
            temperature: Generation temperature (0.0-2.0). Controls
                randomness in generation:
                - 0.0-0.3: Very consistent, minimal variation
                - 0.4-0.7: Balanced creativity and consistency
                - 0.8-1.0: More creative and varied
                - 1.1-2.0: Highly creative but less predictable
                Default 0.7 works well for most improvements.
            provider: Optional LLM provider ("openai", "anthropic", "ollama", etc.)
        """
        self.model = model
        self.temperature = temperature
        self.provider = provider
        self._client: Optional[LLMClient] = None

    async def get_client(self) -> LLMClient:
        """Get or lazily create the LLM client.

        Uses lazy initialization to avoid creating clients until needed.
        The client is cached for reuse across multiple improvements.

        Returns:
            Configured LLMClient instance for the specified model
        """
        if self._client is None:
            self._client = await LLMManager.get_client(
                provider=self.provider, model=self.model, temperature=self.temperature
            )
        return self._client

    async def generate_improvement(
        self, current_text: str, result: SifakaResult, show_prompt: bool = False
    ) -> Tuple[Optional[str], Optional[str], int, float]:
        """Generate an improved version of text based on accumulated feedback.

        This is the main method that synthesizes all feedback into a concrete
        improvement. It builds a comprehensive prompt from validation failures
        and critic suggestions, then generates a new version addressing all issues.

        Args:
            current_text: The current version of text to improve. This is
                what critics evaluated and what needs enhancement.
            result: SifakaResult containing all history including:
                - Validation failures that must be addressed
                - Critic feedback and suggestions
                - Previous iterations for context
                - Metadata from specialized critics
            show_prompt: If True, prints the complete prompt to console
                for debugging. Useful for understanding how feedback
                is being interpreted.

        Returns:
            Tuple containing:
            - improved_text: New version addressing feedback, or None if
              generation failed or produced no meaningful change
            - prompt_used: The exact prompt sent to the LLM (for debugging)
            - tokens_used: Total tokens consumed (prompt + completion)
            - processing_time: Time in seconds for the generation

        Note:
            The method gracefully handles failures by returning None for
            the improved text. The engine will handle retries or termination.

        Example:
            >>> # With debugging enabled
            >>> text, prompt, tokens, time = await generator.generate_improvement(
            ...     current_text="The quick brown fox",
            ...     result=result_with_feedback,
            ...     show_prompt=True  # See exact prompt
            ... )
            >>>
            >>> if text:
            ...     print(f"Improvement generated: {text[:100]}...")
            ... else:
            ...     print("No improvement generated")
        """
        # Build improvement prompt
        prompt = self._build_improvement_prompt(current_text, result)

        if show_prompt:
            print("\n" + "=" * 80)
            print("IMPROVEMENT PROMPT")
            print("=" * 80)
            print(prompt)
            print("=" * 80 + "\n")

        start_time = time.time()
        try:
            # Get client
            client = await self.get_client()

            # For Ollama, use direct completion instead of pydantic_ai agent
            if self.provider == "ollama":
                messages = [
                    {"role": "system", "content": self.IMPROVEMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]

                response = await client.complete(messages)
                processing_time = time.time() - start_time

                # Parse the response as improved text
                improved_text = response.content.strip()
                tokens_used = (
                    response.usage.get("total_tokens", 0) if response.usage else 0
                )

                return improved_text, prompt, tokens_used, processing_time

            # For other providers, use PydanticAI agent for structured output
            if logfire:
                with logfire.span(
                    "text_generation_llm_call",
                    model=self.model,
                    provider=self.provider or "unknown",
                    iteration=result.iteration,
                ) as span:
                    agent = client.create_agent(
                        system_prompt=self.IMPROVEMENT_SYSTEM_PROMPT,
                        result_type=ImprovementResponse,
                    )

                    # Run agent to get structured improvement with usage tracking
                    agent_result = await agent.run(prompt)
                    processing_time = time.time() - start_time

                    improvement = agent_result.output

                    # Get usage data
                    tokens_used = 0
                    try:
                        if hasattr(agent_result, "usage"):
                            usage = agent_result.usage()  # Call as function
                            if usage and hasattr(usage, "total_tokens"):
                                tokens_used = getattr(usage, "total_tokens", 0)
                    except Exception:
                        # Fallback if usage() call fails
                        tokens_used = 0

                    # Log metrics to span
                    span.set_attribute("llm.tokens_used", tokens_used)
                    span.set_attribute(
                        "llm.duration_seconds", round(processing_time, 3)
                    )
                    span.set_attribute("llm.prompt_length", len(prompt))
            else:
                agent = client.create_agent(
                    system_prompt=self.IMPROVEMENT_SYSTEM_PROMPT,
                    result_type=ImprovementResponse,
                )

                # Run agent to get structured improvement with usage tracking
                agent_result = await agent.run(prompt)
                processing_time = time.time() - start_time

                improvement = agent_result.output

                # Get usage data
                tokens_used = 0
                try:
                    if hasattr(agent_result, "usage"):
                        usage = agent_result.usage()  # Call as function
                        if usage and hasattr(usage, "total_tokens"):
                            tokens_used = getattr(usage, "total_tokens", 0)
                except Exception:
                    # Fallback if usage() call fails
                    tokens_used = 0

            # Handle case where improvement is a string
            if isinstance(improvement, str):
                # If it's just a string, that's the improved text
                if improvement and improvement != current_text:
                    return improvement, prompt, tokens_used, processing_time
                else:
                    return None, prompt, tokens_used, processing_time

            # Validate improvement for structured response
            improved_text = getattr(improvement, "improved_text", None)  # type: ignore[unreachable]
            if not improved_text or improved_text == current_text:
                return None, prompt, tokens_used, processing_time

            return improved_text, prompt, tokens_used, processing_time

        except ValueError as e:
            # Re-raise ValueError (e.g., no API key) with clear message
            if "No API key found" in str(e):
                raise ModelProviderError(
                    f"Cannot improve text: {str(e)}",
                    provider="unknown",
                    error_code="authentication",
                ) from e
            raise
        except Exception:
            # Log error but return None to allow graceful degradation
            processing_time = time.time() - start_time
            return None, prompt, 0, processing_time

    def _build_improvement_prompt(self, text: str, result: SifakaResult) -> str:
        """Build a comprehensive prompt for text improvement.

        Constructs a carefully structured prompt that includes:
        1. Clear instructions for improvement
        2. The current text to improve
        3. Validation failures (highest priority)
        4. Critic feedback (organized by critic)
        5. Specific improvement instructions

        The prompt is designed to be clear, actionable, and focused
        on addressing all identified issues.

        Args:
            text: Current text to improve
            result: Result object with all feedback

        Returns:
            Complete prompt string ready for LLM
        """
        prompt_parts = [
            "Please improve the following text based on the feedback provided.",
            f"\nCurrent text:\n{text}\n",
        ]

        # Add validation feedback
        if result.validations:
            validation_feedback = self._format_validation_feedback(result)
            if validation_feedback:
                prompt_parts.append(f"\nValidation issues:\n{validation_feedback}\n")

        # Add critique feedback
        if result.critiques:
            critique_feedback = self._format_critique_feedback(result)
            if critique_feedback:
                prompt_parts.append(f"\nCritic feedback:\n{critique_feedback}\n")

        # Add improvement instructions
        prompt_parts.append(
            "\nProvide an improved version that addresses all feedback while "
            "maintaining the original intent. Return only the improved text."
        )

        return "".join(prompt_parts)

    def _format_validation_feedback(self, result: SifakaResult) -> str:
        """Format validation failures into clear requirements.

        Validation failures are the highest priority - the improved text
        MUST pass all validations. This method formats them clearly.

        Args:
            result: Result containing validation history

        Returns:
            Formatted string of validation requirements, empty if all pass

        Note:
            Only includes failed validations from recent attempts to
            avoid cluttering the prompt with outdated issues.
        """
        feedback_lines = []

        # Get recent validations
        recent_validations = list(result.validations)[-5:]

        for validation in recent_validations:
            if not validation.passed:
                feedback_lines.append(f"- {validation.validator}: {validation.details}")

        return "\n".join(feedback_lines)

    def _format_critique_feedback(self, result: SifakaResult) -> str:
        """Format critic feedback into actionable improvements.

        Organizes feedback from multiple critics into a clear structure,
        avoiding duplication and prioritizing specific suggestions.
        Includes critic-specific metadata when it provides actionable
        guidance.

        Args:
            result: Result containing critique history

        Returns:
            Formatted string of all relevant critic feedback

        Design choices:
        - Limits to 5 most recent critiques to prevent prompt bloat
        - Deduplicates feedback from the same critic
        - Includes up to 3 suggestions per critic
        - Integrates critic-specific metadata when helpful
        """
        feedback_lines = []

        # Get recent critiques
        recent_critiques = list(result.critiques)[-5:]

        # Track which critics we've already included to avoid duplication
        included_critics = set()

        for critique in recent_critiques:
            if critique.needs_improvement and critique.critic not in included_critics:
                # Mark this critic as included
                included_critics.add(critique.critic)

                # Add main feedback
                feedback_lines.append(f"\n{critique.critic}:")
                feedback_lines.append(f"- {critique.feedback}")

                # Add specific suggestions
                if critique.suggestions:
                    feedback_lines.append("  Suggestions:")
                    for suggestion in critique.suggestions[:3]:
                        feedback_lines.append(f"  * {suggestion}")

                # Add critic-specific insights from metadata
                if critique.metadata:
                    self._add_critic_insights(critique, feedback_lines)

        return "\n".join(feedback_lines)

    def _add_critic_insights(
        self, critique: "CritiqueResult", lines: List[str]
    ) -> None:
        """Extract and format critic-specific insights from metadata.

        Different critics provide specialized metadata that can guide
        improvements. This method extracts actionable insights while
        avoiding information overload.

        Args:
            critique: Critique result potentially containing metadata
            lines: List to append formatted insights to

        Critic-specific handling:
        - self_rag: Retrieval opportunities for missing information
        - self_refine: Specific refinement targets
        - n_critics: Consensus warnings for major disagreements

        Design principle:
        Only include metadata that directly helps improve the text.
        Diagnostic information is excluded to keep prompts focused.
        """
        metadata = critique.metadata
        if not metadata:
            return

        # SelfRAG: Add specific retrieval needs
        if critique.critic == "self_rag" and "retrieval_opportunities" in metadata:
            opps = metadata.get("retrieval_opportunities", [])
            if opps:
                lines.append("  Information needed:")
                for opp in opps[:3]:
                    if isinstance(opp, dict) and opp.get("reason"):
                        lines.append(f"  - {opp.get('reason', '')}")

        # SelfRefine: Add specific refinement targets
        elif critique.critic == "self_refine" and "refinement_areas" in metadata:
            areas = metadata.get("refinement_areas", [])
            for area in areas[:3]:
                if isinstance(area, dict) and area.get("target_state"):
                    lines.append(f"  - Refine to: {area.get('target_state', '')}")

        # NCritics: Add consensus warning if very low
        elif critique.critic == "n_critics" and "consensus_score" in metadata:
            consensus = metadata.get("consensus_score", 0)
            if consensus < 0.3:
                lines.append(
                    f"  ⚠️ Very low consensus ({consensus:.1f}) - major disagreement between perspectives"
                )
