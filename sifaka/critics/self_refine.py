"""Self-Refine critic implementation.

Based on: Self-Refine: Iterative Refinement with Self-Feedback
Paper: https://arxiv.org/abs/2303.17651
Authors: Madaan et al. (2023)

Self-Refine enables models to iteratively improve their outputs
through self-generated feedback and refinement.

## Similarity to Original Paper:
- DIRECT: Most faithful implementation to the paper
- PRESERVED: Iterative self-improvement without external feedback
- PRESERVED: Model critiques its own output
- PRESERVED: No additional training required

## Implementation Choices:
1. 5 evaluation dimensions: clarity, content, structure, engagement, completeness
2. Quality score threshold of 0.8 to determine if refinement needed
3. Iteration context tracks refinement history
4. Each dimension scored 0.0-1.0 for granular feedback
5. Overall quality score in metadata for refinement decisions

## Why This Approach:
- Self-Refine naturally fits text improvement use case
- Dimensional evaluation provides structured self-feedback
- Iteration tracking helps avoid repetitive suggestions
- Quality threshold prevents over-refinement
- No external data or models needed - pure self-improvement
"""

from typing import Optional, Union

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .base import BaseCritic, CriticConfig, create_prompt_with_format


class SelfRefineCritic(BaseCritic):
    """Implements Self-Refine iterative improvement through self-feedback."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.5,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[CriticConfig] = None,
    ):
        # Initialize with custom config for self-refine
        if config is None:
            config = CriticConfig(
                response_format="json",
                base_confidence=0.6,
                context_weight=0.2,  # Higher weight for iterative context
                depth_weight=0.15,
                domain_weight=0.05,
            )
        super().__init__(model, temperature, config, provider, api_key)

    @property
    def name(self) -> str:
        return "self_refine"

    async def _generate_critique(self, text: str, result: SifakaResult) -> str:
        """Generate self-refinement critique."""
        # Build iteration context
        iteration_context = self._build_iteration_context(result)

        base_prompt = f"""You are using the Self-Refine technique to iteratively improve text.

{iteration_context}

Text to refine:
{text}

Evaluate the text across these dimensions:
1. Clarity and readability
2. Content quality and accuracy
3. Structure and organization
4. Engagement and flow
5. Completeness

Provide specific, actionable feedback for refinement. Focus on what can be improved in the next iteration.
Rate the overall quality (0.0-1.0) and identify specific areas for refinement."""

        prompt = create_prompt_with_format(
            base_prompt, self.config.response_format, include_examples=True
        )

        response = await self.client.complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert text refiner using the Self-Refine technique.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        return response.content

    def _build_iteration_context(self, result: SifakaResult) -> str:
        """Build context from iteration history."""
        if result.iteration == 0:
            return "This is the first iteration. Focus on identifying all areas for improvement."

        context_parts = [f"This is iteration {result.iteration + 1} of refinement."]

        # Include last critique if available
        if result.critiques:
            last_critique = list(result.critiques)[-1]
            context_parts.append(
                f"\nPrevious feedback: {last_critique.feedback[:200]}..."
            )
            if last_critique.suggestions:
                context_parts.append(
                    f"Previous key suggestion: {last_critique.suggestions[0]}"
                )

        # Track improvement trajectory
        if len(result.generations) > 1:
            context_parts.append(
                f"\nThe text has been refined {len(result.generations)} times."
            )
            context_parts.append(
                "Focus on whether previous suggestions were addressed and identify remaining issues."
            )

        return "\n".join(context_parts)

    def _parse_json_response(self, response: str) -> "CriticResponse":
        """Parse JSON response with self-refine specific handling."""
        try:
            critic_response = super()._parse_json_response(response)

            # Extract quality score from metadata if available
            quality_score = critic_response.metadata.get("quality_score", None)
            if quality_score is not None:
                # Adjust confidence based on quality score
                critic_response.confidence = min(0.9, quality_score + 0.1)

                # Override needs_improvement based on quality
                if quality_score < 0.8:
                    critic_response.needs_improvement = True

            # Self-refine specific: more suggestions usually means more refinement needed
            if len(critic_response.suggestions) > 3:
                critic_response.needs_improvement = True
                critic_response.confidence *= (
                    0.9  # Slightly lower confidence with many issues
                )

            return critic_response

        except Exception:
            # Fallback to standard parsing
            return super()._parse_json_response(response)
