"""Constitutional AI critic implementation.

Based on: Constitutional AI: Harmlessness from AI Feedback
Paper: https://arxiv.org/abs/2212.08073
Authors: Bai et al. (2022)

Constitutional AI uses a set of principles to guide AI behavior toward
being helpful, harmless, and honest.

## Similarity to Original Paper:
- PRESERVED: Use of constitutional principles for evaluation
- PRESERVED: Focus on harmlessness and helpfulness
- ENHANCED: Two-stage process with self-critique and revision
- PRESERVED: AI feedback to improve outputs
- ADAPTED: Principles focused on text quality + safety

## Implementation Choices:
1. Two-stage process: critique then revise
2. Self-critique of generated improvements
3. Severity scoring for principle violations
4. HHH (Helpful, Harmless, Honest) framework
5. Configurable principles via Config

## Why This Approach:
- Matches paper's self-improvement concept
- Provides both critique and solutions
- Clear criteria with actionable outputs
- Balances safety with quality
"""

from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from ..core.config import Config


# Default constitutional principles
DEFAULT_PRINCIPLES = [
    "Be helpful and constructive in feedback",
    "Ensure content is safe and appropriate for all audiences",
    "Promote accuracy and avoid misinformation",
    "Encourage clarity and good communication",
    "Respect diverse perspectives and avoid bias",
    "Maintain professional and respectful tone",
]


class PrincipleEvaluation(BaseModel):
    """Evaluation of a single constitutional principle."""

    principle: str = Field(..., description="The principle being evaluated")
    category: str = Field(..., description="Category: helpful, harmless, or honest")
    passed: bool = Field(..., description="Whether the text passes this principle")
    severity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Violation severity (0=none, 1=severe)"
    )
    violations: list[str] = Field(
        default_factory=list, description="Specific violations found"
    )
    improvements: list[str] = Field(
        default_factory=list, description="How to better align with this principle"
    )


class RevisionProposal(BaseModel):
    """A proposed revision to address constitutional issues."""

    original_snippet: str = Field(..., description="The problematic part of the text")
    revised_snippet: str = Field(..., description="The constitutional revision")
    principles_addressed: list[str] = Field(
        ..., description="Which principles this revision addresses"
    )
    improvement_rationale: str = Field(..., description="Why this revision is better")


class ConstitutionalResponse(BaseModel):
    """Response model specific to Constitutional AI critic."""

    feedback: str = Field(..., description="Overall constitutional assessment")
    suggestions: list[str] = Field(
        default_factory=list,
        description="Specific improvements to align with principles",
    )
    needs_improvement: bool = Field(
        ..., description="Whether the text violates any constitutional principles"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in constitutional assessment",
    )

    # Constitutional AI specific fields
    principle_evaluations: list[PrincipleEvaluation] = Field(
        default_factory=list, description="Detailed evaluation per principle"
    )
    revision_proposals: list[RevisionProposal] = Field(
        default_factory=list, description="Specific revisions to address violations"
    )

    # HHH scores
    helpfulness_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="How helpful is the content"
    )
    harmlessness_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="How harmless/safe is the content"
    )
    honesty_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="How honest/accurate is the content"
    )

    # Meta fields
    requires_major_revision: bool = Field(
        default=False, description="Whether violations require substantial rewriting"
    )
    self_critique_notes: str = Field(
        default="", description="Self-critique of the proposed improvements"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional constitutional data"
    )


class ConstitutionalCritic(BaseCritic):
    """Implements Constitutional AI principles for text evaluation.

    When to Use This Critic:
    - ✅ Safety and accuracy are paramount
    - ✅ You have specific principles/guidelines to follow
    - ✅ Ethical considerations matter
    - ✅ Need consistent principle-based evaluation
    - ❌ Creative freedom is the priority
    - ❌ Principles might constrain innovation
    - 🎯 Best for: Public content, educational material, compliance docs
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        principles: Optional[List[str]] = None,
    ):
        # Initialize with custom config
        if config is None:
            config = Config()

        super().__init__(model, temperature, config, provider, api_key)

        # Use provided principles or config principles or defaults
        self.principles = (
            principles
            or (config.constitutional_principles if config else None)
            or DEFAULT_PRINCIPLES
        )

    @property
    def name(self) -> str:
        return "constitutional"

    def _get_response_type(self) -> type[BaseModel]:
        """Use custom ConstitutionalResponse for structured output."""
        return ConstitutionalResponse

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for constitutional evaluation."""
        # Format principles
        principles_text = "\n".join(
            f"{i+1}. {principle}" for i, principle in enumerate(self.principles)
        )

        # Get previous context
        previous_context = self._get_previous_context(result)

        user_prompt = f"""Apply Constitutional AI's two-stage evaluation and revision process.

Constitutional Principles:
{principles_text}

Text to evaluate:
{text}
{previous_context}

Stage 1 - Constitutional Evaluation:
1. Assess each principle for violations (categorize as Helpful, Harmless, or Honest)
2. Score violation severity (0.0 = none, 1.0 = severe)
3. Identify specific problematic content

Stage 2 - Constitutional Revision:
1. For each violation, propose a specific revision
2. Explain how the revision addresses the principle
3. Self-critique: Are the revisions themselves constitutional?
4. Ensure revisions maintain helpfulness while improving harmlessness/honesty

Provide HHH scores and determine if major revision is needed."""

        return [
            {
                "role": "system",
                "content": "You are a Constitutional AI critic that evaluates and revises text using the HHH framework (Helpful, Harmless, Honest). You both critique violations and propose constitutional revisions, then self-critique those revisions.",
            },
            {"role": "user", "content": user_prompt},
        ]
