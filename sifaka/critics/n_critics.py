"""N-Critics ensemble critic implementation.

Based on: N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics
Paper: https://arxiv.org/abs/2310.18679
Authors: Tian et al. (2023)

N-Critics uses multiple critical perspectives to provide comprehensive
text evaluation through ensemble critique.

## Similarity to Original Paper:
- PRESERVED: Multiple critical perspectives for evaluation
- PRESERVED: Ensemble approach to critique
- SIMPLIFIED: Perspectives instead of separate critic models
- PRESERVED: Consensus building from multiple viewpoints

## Implementation Choices:
1. 4 default perspectives: Clarity, Accuracy, Completeness, Style
2. Each perspective gets independent assessment and score
3. Consensus score from averaging perspective scores
4. Agreement factor from score variance (low variance = high agreement)
5. Confidence combines consensus score (70%) + agreement (30%)

## Why This Approach:
- Single model with multiple prompting perspectives is more efficient
- Preserves multi-viewpoint evaluation without multiple models
- Customizable perspectives for different domains
- Variance-based agreement naturally measures consensus
- Simpler than training/managing multiple critic models
"""

from typing import List, Optional, Dict, Any, Union
import openai

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .base import BaseCritic, CriticConfig, create_prompt_with_format


class NCriticsCritic(BaseCritic):
    """Implements N-Critics ensemble approach for comprehensive evaluation."""

    # Default critical perspectives
    DEFAULT_PERSPECTIVES = [
        "Clarity: Focus on readability, structure, and comprehensibility",
        "Accuracy: Focus on factual correctness and logical consistency",
        "Completeness: Focus on thoroughness and adequate coverage",
        "Style: Focus on tone, voice, and appropriateness for audience",
    ]

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.6,
        perspectives: Optional[List[str]] = None,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[CriticConfig] = None,
    ):
        # Initialize with custom config for n-critics
        if config is None:
            config = CriticConfig(
                response_format="json",
                base_confidence=0.65,
                context_weight=0.1,
                depth_weight=0.15,
                domain_weight=0.1,
            )
        super().__init__(model, temperature, config, provider=provider, api_key=api_key)
        self.perspectives = perspectives or self.DEFAULT_PERSPECTIVES

    @property
    def name(self) -> str:
        return "n_critics"

    async def _generate_critique(self, text: str, result: SifakaResult) -> str:
        """Generate ensemble critique from multiple perspectives."""
        perspectives_text = "\n".join(
            f"{i+1}. {perspective}" for i, perspective in enumerate(self.perspectives)
        )

        base_prompt = f"""Using the N-Critics ensemble technique, evaluate this text from multiple critical perspectives:

CRITICAL PERSPECTIVES:
{perspectives_text}

Text to evaluate:
{text}

For EACH perspective, provide:
1. A focused assessment from that viewpoint
2. Specific strengths and weaknesses
3. Targeted suggestions for improvement

Then provide an overall consensus assessment synthesizing all perspectives."""

        # Add specific format for JSON response
        if self.config.response_format == "json":
            prompt = (
                base_prompt
                + """

Provide your response in this JSON format:
{
    "feedback": "Overall ensemble assessment synthesizing all perspectives",
    "suggestions": ["Suggestion from perspective 1", "Suggestion from perspective 2", ...],
    "needs_improvement": true/false,
    "confidence": 0.0-1.0,
    "metadata": {
        "perspective_assessments": {
            "1": {"assessment": "...", "score": 0.0-1.0},
            "2": {"assessment": "...", "score": 0.0-1.0},
            ...
        },
        "consensus_score": 0.0-1.0
    }
}"""
            )
        else:
            prompt = create_prompt_with_format(base_prompt, self.config.response_format)

        response = await self.client.complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert text critic using the N-Critics ensemble technique.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        return response.content

    def _parse_json_response(self, response: str) -> "CriticResponse":
        """Parse JSON response with N-Critics specific handling."""
        try:
            critic_response = super()._parse_json_response(response)

            # Extract perspective assessments and consensus
            metadata = critic_response.metadata
            perspective_assessments = metadata.get("perspective_assessments", {})
            consensus_score = metadata.get("consensus_score", 0.7)

            # Calculate ensemble confidence based on perspective agreement
            if perspective_assessments:
                scores = [p.get("score", 0.5) for p in perspective_assessments.values()]
                if scores:
                    # High agreement = high confidence
                    avg_score = sum(scores) / len(scores)
                    variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
                    agreement_factor = 1.0 - min(
                        variance * 2, 0.5
                    )  # Lower variance = higher agreement

                    # Combine consensus score with agreement
                    critic_response.confidence = min(
                        0.9, consensus_score * 0.7 + agreement_factor * 0.3
                    )
            else:
                # Use consensus score directly if no detailed assessments
                critic_response.confidence = min(0.9, consensus_score + 0.1)

            # Override needs_improvement based on consensus
            if consensus_score < 0.75:
                critic_response.needs_improvement = True

            # Ensure we have suggestions from each perspective
            if len(critic_response.suggestions) < len(self.perspectives):
                # Add a generic suggestion if missing
                critic_response.suggestions.append(
                    "Consider all critical perspectives for improvement"
                )

            return critic_response

        except Exception:
            # Fallback to standard parsing
            return super()._parse_json_response(response)
