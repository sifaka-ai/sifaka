"""N-Critics ensemble critic implementation for multi-perspective evaluation.

Based on: N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics
Paper: https://arxiv.org/abs/2310.18679
Authors: Tian et al. (2023)

N-Critics uses multiple critical perspectives to provide comprehensive
text evaluation through ensemble critique, capturing diverse viewpoints
that single-perspective critics might miss.

## Similarity to Original Paper:

- PRESERVED: Multiple critical perspectives for comprehensive evaluation
- PRESERVED: Ensemble approach to critique generation
- PRESERVED: Consensus building from multiple expert viewpoints
- SIMPLIFIED: Single model with multiple perspectives vs. separate critic models
- ENHANCED: Customizable perspectives for domain-specific evaluation

## Implementation Strategy:

1. **Multi-Perspective Framework**: Uses 4 default critical perspectives:
   - **Clarity**: Readability, structure, and comprehensibility assessment
   - **Accuracy**: Factual correctness and logical consistency evaluation
   - **Completeness**: Thoroughness and adequate coverage analysis
   - **Style**: Tone, voice, and audience appropriateness review

2. **Ensemble Evaluation**: Each perspective provides independent assessment
   and scoring, then results are synthesized into comprehensive feedback

3. **Consensus Scoring**: Overall quality score derived from averaging
   perspective assessments, with confidence based on agreement levels

4. **Adaptive Perspectives**: Optional auto-generation of context-appropriate
   perspectives based on text type and domain

## Why This Approach:

- **Comprehensive Coverage**: Multiple perspectives catch issues single critics miss
- **Reduced Bias**: Ensemble approach mitigates individual perspective limitations
- **Efficiency**: Single model with multiple prompts vs. multiple model instances
- **Flexibility**: Customizable perspectives for different content types
- **Quality Assurance**: Consensus scoring provides confidence metrics
- **Scalability**: Works across various domains and content types

## Consensus Mechanism:

The critic builds consensus by:
1. Evaluating text from each perspective independently
2. Synthesizing feedback that addresses all viewpoints
3. Generating consensus score based on perspective agreement
4. Providing confidence metrics based on inter-perspective consistency

This approach is particularly valuable for complex content requiring
comprehensive evaluation from multiple expert angles.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..core.config import Config
from ..core.llm_client import Provider
from ..core.models import SifakaResult
from .core.base import BaseCritic

# Removed PerspectiveAssessment class since it's not used in generation.py


class NCriticsResponse(BaseModel):
    """Structured response model for N-Critics ensemble critique.

    Provides synthesized feedback from multiple critical perspectives,
    including consensus scoring and confidence metrics based on
    inter-perspective agreement.

    Attributes:
        feedback: Synthesized feedback combining insights from all
            critical perspectives into comprehensive assessment.
        suggestions: Combined improvement recommendations addressing
            issues identified across multiple viewpoints.
        needs_improvement: Consensus decision on whether text requires
            improvement based on multi-perspective evaluation.
        confidence: Confidence level based on agreement between perspectives
            (higher when perspectives align, lower with disagreement).
        consensus_score: Overall quality score (0.0-1.0) representing
            synthesized assessment across all critical perspectives.
    """

    feedback: str = Field(..., description="Synthesized feedback from all perspectives")
    suggestions: list[str] = Field(
        default_factory=list, description="Combined suggestions from all perspectives"
    )
    needs_improvement: bool = Field(
        ..., description="Consensus on whether improvement is needed"
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence based on perspective agreement",
    )
    # Only keep consensus_score since that's what's used in generation.py
    consensus_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Overall consensus score from all perspectives",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional n-critics data"
    )


class NCriticsCritic(BaseCritic):
    """Implements N-Critics ensemble approach for multi-perspective evaluation.

    Provides comprehensive text assessment using multiple critical perspectives
    that work together to identify issues, strengths, and improvements that
    single-perspective critics might miss.

    ## When to Use This Critic:

    ✅ **Ideal for:**
    - Complex, multifaceted content requiring comprehensive review
    - High-stakes documents needing thorough quality assessment
    - Content where multiple expert viewpoints add value
    - Situations where single critics might have blind spots
    - Final quality assurance before important publications
    - Content spanning multiple domains or audiences

    ❌ **Avoid when:**
    - Quick, single-dimension quality checks
    - Simple content where perspectives would be redundant
    - Resource-constrained environments (higher token usage)
    - Time-sensitive evaluations needing rapid feedback
    - Content with narrow, specialized focus

    🎯 **Optimal applications:**
    - Comprehensive document reviews (reports, proposals, papers)
    - Multi-stakeholder content (presentations, communications)
    - Complex technical documentation requiring multiple viewpoints
    - Editorial reviews for publication-quality content
    - Cross-functional content needing diverse expert perspectives
    - Quality assurance for customer-facing materials

    ## Perspective Framework:

    **Default Perspectives:**
    1. **Clarity**: Readability, structure, comprehensibility
    2. **Accuracy**: Factual correctness, logical consistency
    3. **Completeness**: Thoroughness, adequate coverage
    4. **Style**: Tone, voice, audience appropriateness

    **Custom Perspectives:**
    Can be tailored for specific domains:
    - Technical writing: Accuracy, Completeness, Usability, Maintainability
    - Marketing: Persuasiveness, Brand alignment, Clarity, Call-to-action
    - Academic: Rigor, Methodology, Citations, Contribution

    ## Consensus Building:

    The critic synthesizes perspectives by:
    - Evaluating text independently from each viewpoint
    - Identifying areas of agreement and disagreement
    - Building consensus feedback that addresses all perspectives
    - Providing confidence metrics based on inter-perspective consistency

    ## Value Proposition:

    This approach is particularly valuable when:
    - Multiple expert opinions would naturally be sought
    - Content complexity benefits from diverse analytical angles
    - Quality requirements demand comprehensive evaluation
    - Risk of oversight in single-perspective assessment is high
    """

    # Default critical perspectives for comprehensive evaluation
    DEFAULT_PERSPECTIVES = [
        "Clarity: Focus on readability, structure, and comprehensibility",
        "Accuracy: Focus on factual correctness and logical consistency",
        "Completeness: Focus on thoroughness and adequate coverage",
        "Style: Focus on tone, voice, and appropriateness for audience",
    ]
    """Default ensemble of critical perspectives.

    These four perspectives provide comprehensive coverage for most content types:
    - Clarity ensures the content is understandable and well-organized
    - Accuracy verifies factual correctness and logical flow
    - Completeness checks for adequate coverage and thoroughness
    - Style evaluates appropriateness for audience and purpose

    Can be customized for domain-specific evaluation needs.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.6,
        perspectives: Optional[List[str]] = None,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        auto_generate_perspectives: bool = False,
        perspective_count: int = 4,
    ):
        """Initialize N-Critics ensemble critic with multiple perspectives.

        Creates a comprehensive critic that evaluates text from multiple
        expert viewpoints to provide thorough, well-rounded feedback.

        Args:
            model: LLM model for multi-perspective evaluation. GPT-4o-mini
                provides good balance for ensemble evaluation tasks.
            temperature: Generation temperature (0.0-1.0). Moderate values
                (0.5-0.7) balance consistency across perspectives with insight.
            perspectives: Custom list of critical perspectives. If None,
                uses DEFAULT_PERSPECTIVES. Each should specify focus area.
            provider: LLM provider (OpenAI, Anthropic, etc.)
            api_key: API key override if not using environment variables
            config: Full Sifaka configuration object
            auto_generate_perspectives: If True, generates context-appropriate
                perspectives based on text content (experimental feature).
            perspective_count: Number of perspectives when auto-generating.
                Only used if auto_generate_perspectives=True.

        Example:
            >>> # Use default perspectives
            >>> critic = NCriticsCritic()
            >>>
            >>> # Custom perspectives for technical writing
            >>> tech_perspectives = [
            ...     "Accuracy: Technical correctness and precision",
            ...     "Usability: Ease of implementation and use",
            ...     "Completeness: Coverage of necessary information",
            ...     "Clarity: Accessibility to target audience"
            ... ]
            >>> critic = NCriticsCritic(perspectives=tech_perspectives)
            >>>
            >>> # Auto-generate context-specific perspectives
            >>> critic = NCriticsCritic(auto_generate_perspectives=True)

        Perspective format:
            Each perspective should be formatted as "Name: Description"
            where the description guides the evaluation focus.

        Resource considerations:
            Ensemble evaluation uses more tokens than single-perspective
            critics due to multi-viewpoint analysis.
        """
        # Initialize with custom config
        if config is None:
            config = Config()
        super().__init__(model, temperature, config, provider, api_key)

        self.auto_generate_perspectives = auto_generate_perspectives
        self.perspective_count = perspective_count
        self.perspectives = perspectives or self.DEFAULT_PERSPECTIVES

    @property
    def name(self) -> str:
        """Return the unique identifier for this critic.

        Returns:
            "n_critics" - used in configuration, logging, and metadata
        """
        return "n_critics"

    def _get_response_type(self) -> type[BaseModel]:
        """Specify the structured response format for ensemble critique.

        Returns:
            NCriticsResponse class providing synthesized multi-perspective
            feedback with consensus scoring and confidence metrics.
        """
        return NCriticsResponse

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create comprehensive messages for multi-perspective ensemble critique.

        Builds detailed instructions for evaluating text from multiple critical
        perspectives, synthesizing insights, and building consensus feedback.

        Args:
            text: Text to evaluate from multiple critical perspectives
            result: SifakaResult containing context and evaluation history

        Returns:
            List of message dictionaries instructing the LLM to perform
            comprehensive multi-perspective evaluation with consensus building

        Note:
            If auto_generate_perspectives is enabled, this method will first
            generate context-appropriate perspectives before evaluation.
        """
        # Generate perspectives if requested
        if self.auto_generate_perspectives:
            self.perspectives = await self._generate_perspectives(text, result)

        perspectives_text = "\n".join(
            f"{i + 1}. {perspective}" for i, perspective in enumerate(self.perspectives)
        )

        # Get previous context
        previous_context = self._get_previous_context(result)

        user_prompt = f"""Using the N-Critics ensemble technique, evaluate this text from multiple critical perspectives:

CRITICAL PERSPECTIVES:
{perspectives_text}

Text to evaluate:
{text}
{previous_context}

Evaluate the text from each perspective and provide:
- Overall synthesized feedback combining all perspectives
- Specific improvement suggestions that address issues from multiple viewpoints
- A consensus score (0.0-1.0) representing overall text quality across all perspectives

Focus on constructive, actionable feedback that considers all viewpoints."""

        return [
            {
                "role": "system",
                "content": "You are an expert text critic using the N-Critics ensemble technique to evaluate text from multiple critical perspectives. Provide balanced, comprehensive feedback.",
            },
            {"role": "user", "content": user_prompt},
        ]

    async def _generate_perspectives(
        self, text: str, result: SifakaResult
    ) -> List[str]:
        """Generate context-appropriate perspectives for the text.

        Analyzes the text content and context to dynamically generate
        the most relevant critical perspectives for evaluation.

        Args:
            text: Text content to analyze for perspective generation
            result: SifakaResult providing additional context

        Returns:
            List of generated perspectives tailored to the content type
            and evaluation context

        Note:
            This is an experimental feature. Currently returns default
            perspectives, but could be enhanced to use LLM analysis
            for dynamic perspective generation based on content type,
            domain, audience, and purpose.

        Future enhancements:
            - Content type detection (technical, marketing, academic, etc.)
            - Domain-specific perspective libraries
            - Audience-appropriate evaluation criteria
            - Purpose-driven perspective selection
        """
        # TODO: Implement LLM-based perspective generation
        # This would analyze text type, domain, audience to generate
        # context-appropriate perspectives for evaluation
        return self.DEFAULT_PERSPECTIVES
