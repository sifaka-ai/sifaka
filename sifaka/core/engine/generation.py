"""Text generation component of the Sifaka engine."""

import time
from typing import Optional, Tuple, List
from pydantic import BaseModel, Field

from ..models import SifakaResult, CritiqueResult
from ..llm_client import LLMManager, LLMClient


class ImprovementResponse(BaseModel):
    """Structured response for text improvements."""

    improved_text: str = Field(..., description="The improved version of the text")
    changes_made: list[str] = Field(
        default_factory=list, description="List of changes made"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in improvements"
    )


class TextGenerator:
    """Handles text generation and improvement."""

    IMPROVEMENT_SYSTEM_PROMPT = """You are an expert text editor focused on iterative improvement. Pay careful attention to all critic feedback and validation issues. Your goal is to address each piece of feedback thoroughly while maintaining the original intent and improving the overall quality of the text."""

    def __init__(self, model: str, temperature: float):
        """Initialize text generator.

        Args:
            model: LLM model to use
            temperature: Generation temperature
        """
        self.model = model
        self.temperature = temperature
        self._client: Optional[LLMClient] = None

    @property
    def client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._client is None:
            self._client = LLMManager.get_client(
                model=self.model, temperature=self.temperature
            )
        return self._client

    async def generate_improvement(
        self, current_text: str, result: SifakaResult, show_prompt: bool = False
    ) -> Tuple[Optional[str], Optional[str], int, float]:
        """Generate improved text based on feedback.

        Args:
            current_text: Current version of text
            result: Result object with critique history
            show_prompt: Whether to print the prompt

        Returns:
            Tuple of (improved_text, prompt_used, tokens_used, processing_time)
        """
        # Build improvement prompt
        prompt = self._build_improvement_prompt(current_text, result)

        if show_prompt:
            print("\n" + "=" * 80)
            print("IMPROVEMENT PROMPT")
            print("=" * 80)
            print(prompt)
            print("=" * 80 + "\n")

        try:
            # Create PydanticAI agent for structured improvement
            agent = self.client.create_agent(
                system_prompt=self.IMPROVEMENT_SYSTEM_PROMPT,
                result_type=ImprovementResponse,
            )

            # Run agent to get structured improvement with usage tracking
            start_time = time.time()
            agent_result = await agent.run(prompt)
            processing_time = time.time() - start_time

            improvement = agent_result.output

            # Get usage data
            tokens_used = 0
            try:
                if hasattr(agent_result, "usage"):
                    usage = agent_result.usage()  # Call as function
                    if usage and hasattr(usage, "total_tokens"):
                        tokens_used = usage.total_tokens
            except Exception:
                # Fallback if usage() call fails
                tokens_used = 0

            # Validate improvement
            if (
                not improvement.improved_text
                or improvement.improved_text == current_text
            ):
                return None, prompt, tokens_used, processing_time

            return improvement.improved_text, prompt, tokens_used, processing_time

        except Exception:
            # Return None on error, let engine handle it
            return None, prompt, 0, 0.0

    def _build_improvement_prompt(self, text: str, result: SifakaResult) -> str:
        """Build prompt for text improvement."""
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
        """Format validation feedback for prompt."""
        feedback_lines = []

        # Get recent validations
        recent_validations = list(result.validations)[-5:]

        for validation in recent_validations:
            if not validation.passed:
                feedback_lines.append(f"- {validation.validator}: {validation.details}")

        return "\n".join(feedback_lines)

    def _format_critique_feedback(self, result: SifakaResult) -> str:
        """Format critique feedback for prompt."""
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
        """Add critic-specific insights from metadata.

        Only add metadata that meaningfully helps the next iteration.
        If metadata won't improve the next generation, don't include it.
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
