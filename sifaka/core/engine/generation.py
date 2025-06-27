"""Text generation component of the Sifaka engine."""

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
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate improved text based on feedback.

        Args:
            current_text: Current version of text
            result: Result object with critique history
            show_prompt: Whether to print the prompt

        Returns:
            Tuple of (improved_text, prompt_used)
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

            # Run agent to get structured improvement
            agent_result = await agent.run(prompt)
            improvement = agent_result.output

            # Validate improvement
            if (
                not improvement.improved_text
                or improvement.improved_text == current_text
            ):
                return None, prompt

            return improvement.improved_text, prompt

        except Exception:
            # Return None on error, let engine handle it
            return None, prompt

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

        for critique in recent_critiques:
            if critique.needs_improvement:
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
        """Add critic-specific insights from metadata."""
        metadata = critique.metadata

        # SelfRAG: Add reflection assessments and retrieval opportunities
        if critique.critic == "self_rag":
            if "overall_relevance" in metadata and not metadata.get(
                "overall_relevance"
            ):
                lines.append("  ⚠️ Content relevance issues detected (ISREL)")
            if "overall_support" in metadata and not metadata.get("overall_support"):
                lines.append("  ⚠️ Evidence support needed (ISSUP)")
            if "retrieval_opportunities" in metadata:
                opps = metadata.get("retrieval_opportunities", [])
                if opps:
                    lines.append("  Retrieval would help with:")
                    for opp in opps[:2]:
                        if isinstance(opp, dict):
                            lines.append(
                                f"  - {opp.get('location', 'Unknown')}: {opp.get('reason', '')}"
                            )

        # SelfRefine: Add refinement areas
        elif critique.critic == "self_refine" and "refinement_areas" in metadata:
            lines.append("  Areas needing refinement:")
            for area in metadata.get("refinement_areas", [])[:3]:
                if isinstance(area, dict) and "area" in area:
                    lines.append(f"  - {area['area']}: {area.get('target_state', '')}")

        # NCritics: Add perspective insights
        elif critique.critic == "n_critics" and "perspective_assessments" in metadata:
            consensus = metadata.get("consensus_score", 0)
            if consensus < 0.5:
                lines.append(
                    f"  Low consensus ({consensus:.1f}) - consider diverse perspectives"
                )

        # Constitutional: Add principle violations and revisions
        elif critique.critic == "constitutional":
            if "revision_proposals" in metadata:
                revisions = metadata.get("revision_proposals", [])
                if revisions:
                    lines.append("  Constitutional revisions available:")
                    for rev in revisions[:2]:
                        if isinstance(rev, dict):
                            lines.append(
                                f"  - {rev.get('original_snippet', '')[:30]}... → {rev.get('revised_snippet', '')[:30]}..."
                            )
            if "requires_major_revision" in metadata and metadata.get(
                "requires_major_revision"
            ):
                lines.append("  ⚠️ Major constitutional revision required")

        # MetaRewarding: Add meta-evaluation insights
        elif critique.critic == "meta_rewarding":
            if "improvement_delta" in metadata:
                delta = metadata.get("improvement_delta", 0)
                if delta > 0:
                    lines.append(f"  ✓ Critique refined with {delta:.1%} improvement")
            if "suggestion_preferences" in metadata:
                prefs = metadata.get("suggestion_preferences", [])
                if prefs and isinstance(prefs[0], dict):
                    lines.append(
                        f"  Top suggestion: {prefs[0].get('suggestion', '')[:50]}..."
                    )
