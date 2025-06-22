"""Constitutional AI critic implementation.

Based on: Constitutional AI: Harmlessness from AI Feedback
Paper: https://arxiv.org/abs/2212.08073
Authors: Bai et al. (Anthropic, 2022)

Constitutional AI uses a set of principles to guide AI behavior
and provide structured feedback for improvement.

## Similarity to Original Paper:
- ADAPTED: Safety principles → text quality principles
- PRESERVED: Principle-based evaluation framework
- PRESERVED: Scoring against multiple constitutional rules
- SIMPLIFIED: No RLHF training; uses prompting instead

## Implementation Choices:
1. 7 default principles covering quality dimensions (vs safety focus)
2. 1-5 scoring per principle (more granular than binary safe/unsafe)
3. Violation tracking with severity levels (1-5)
4. Customizable principles via initialization
5. Confidence based on principle coverage and scores

## Why This Approach:
- Constitutional framework translates perfectly to quality assessment
- Principles make evaluation transparent and consistent
- Scoring system provides nuanced feedback beyond pass/fail
- Easy to customize for domain-specific quality requirements
- Maintains the "constitution guides behavior" concept for text quality
"""

from typing import List, Optional, Dict, Any, Union
import openai
from pydantic import BaseModel, Field

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .base import BaseCritic, CriticConfig, CriticResponse, create_prompt_with_format


class PrincipleViolation(BaseModel):
    """Represents a violation of a constitutional principle."""

    principle_number: int = Field(
        ge=1, description="Which principle was violated (1-based index)"
    )
    principle_text: str = Field(description="The text of the violated principle")
    violation_description: str = Field(
        description="Detailed explanation of how the principle was violated"
    )
    severity: int = Field(
        ge=1, le=5, description="Severity of violation (1=minor, 5=severe)"
    )


class ConstitutionalResponse(CriticResponse):
    """Extended response format for constitutional critic."""

    principle_scores: Dict[int, int] = Field(
        default_factory=dict,
        description="Score for each principle (1-based index -> score 1-5, 5=excellent)",
    )
    violations: List[PrincipleViolation] = Field(
        default_factory=list, description="List of principle violations found"
    )


class ConstitutionalCritic(BaseCritic):
    """Implements Constitutional AI principles for text evaluation."""

    # Default constitutional principles for text quality
    DEFAULT_PRINCIPLES = [
        "The text should be clear and easy to understand",
        "The text should be accurate and factual",
        "The text should be well-structured and organized",
        "The text should be engaging and interesting to read",
        "The text should be appropriate for the intended audience",
        "The text should be complete and comprehensive",
        "The text should avoid harmful, offensive, or misleading content",
    ]

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        principles: Optional[List[str]] = None,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[CriticConfig] = None,
    ):
        # Initialize with custom config for constitutional
        if config is None:
            config = CriticConfig(
                response_format="json",
                base_confidence=0.7,
                context_weight=0.05,
                depth_weight=0.1,
                domain_weight=0.15,
            )
        super().__init__(model, temperature, config, provider=provider, api_key=api_key)
        self.principles = principles or self.DEFAULT_PRINCIPLES

    @property
    def name(self) -> str:
        return "constitutional"

    async def _generate_critique(self, text: str, result: SifakaResult) -> str:
        """Generate constitutional evaluation."""
        base_prompt = f"""You are evaluating text against constitutional principles.

Principles to evaluate against:
{self._format_principles()}

Text to evaluate:
{text}

Evaluate the text against each principle and provide:
1. A score (1-5, where 5 is excellent) for each principle
2. Any violations of principles (with severity 1-5)
3. Overall assessment of constitutional compliance
4. Specific suggestions for improvement

Be thorough and specific in your evaluation."""

        # Add specific format for constitutional response
        if self.config.response_format == "json":
            prompt = (
                base_prompt
                + """

Provide your response in this JSON format:
{
    "feedback": "Overall assessment of constitutional compliance",
    "suggestions": ["Specific suggestion 1", "Specific suggestion 2", ...],
    "needs_improvement": true/false,
    "confidence": 0.0-1.0,
    "metadata": {
        "principle_scores": {
            "1": 5, "2": 4, ...
        },
        "violations": [
            {
                "principle_number": 1,
                "principle_text": "The principle text",
                "violation_description": "How it was violated",
                "severity": 1-5
            }
        ]
    }
}"""
            )
        else:
            prompt = create_prompt_with_format(base_prompt, self.config.response_format)

        response = await self.client.complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator using Constitutional AI principles.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        return response.content

    def _format_principles(self) -> str:
        """Format principles for the prompt."""
        formatted = []
        for i, principle in enumerate(self.principles, 1):
            formatted.append(f"{i}. {principle}")
        return "\n".join(formatted)

    def _parse_json_response(self, response: str) -> "CriticResponse":
        """Parse JSON response with constitutional-specific handling."""
        try:
            critic_response = super()._parse_json_response(response)

            # Extract constitutional-specific metadata
            metadata = critic_response.metadata
            principle_scores = metadata.get("principle_scores", {})
            violations = metadata.get("violations", [])

            # Enhance needs_improvement assessment based on violations and scores
            if not critic_response.needs_improvement:
                # Override if we have severe violations or low scores
                if violations:
                    severe_violations = [
                        v for v in violations if v.get("severity", 0) >= 4
                    ]
                    if severe_violations:
                        critic_response.needs_improvement = True

                if principle_scores:
                    low_scores = [
                        score for score in principle_scores.values() if score <= 2
                    ]
                    if len(low_scores) >= 2:  # Multiple low scores
                        critic_response.needs_improvement = True

            # Enhance confidence based on evaluation completeness
            if principle_scores:
                coverage = len(principle_scores) / len(self.principles)
                critic_response.confidence *= 0.8 + 0.2 * coverage

            return critic_response

        except Exception:
            # Fallback to standard parsing
            return super()._parse_json_response(response)
