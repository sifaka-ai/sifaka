"""Self-RAG critic implementation (Reflection-based version).

Based on: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
Paper: https://arxiv.org/abs/2310.11511
Authors: Asai et al. (2023)

Self-RAG teaches LLMs to adaptively retrieve, generate, and critique text
using special reflection tokens to control behavior.

## Similarity to Original Paper:
- PRESERVED: Self-reflection on content quality
- PRESERVED: Adaptive decision-making about information needs
- PRESERVED: Multiple critique dimensions (relevance, support, utility)
- SIMPLIFIED: No actual retrieval system (identifies where needed)
- ADAPTED: Uses structured outputs instead of reflection tokens

## Implementation Choices:
1. Evaluates ISREL (relevance), ISSUP (support), ISUSE (utility)
2. Identifies retrieval opportunities without performing retrieval
3. Provides confidence scores for factual claims
4. Suggests specific source types needed

## Why This Approach:
- Maintains Self-RAG's core reflection concept
- Works without external retrieval infrastructure
- Provides actionable guidance on information gaps
- Preserves the adaptive, self-critical nature
"""

from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from ..core.config import Config


class FactualClaim(BaseModel):
    """A factual claim with Self-RAG reflection assessment."""

    claim: str = Field(..., description="The specific factual claim")
    isrel: bool = Field(
        ..., description="ISREL: Is this claim relevant to the main topic?"
    )
    issup: bool = Field(..., description="ISSUP: Is this claim supported by evidence?")
    isuse: bool = Field(..., description="ISUSE: Is this claim useful for the reader?")
    confidence_level: str = Field(..., description="Confidence: high, medium, low")
    retrieval_needed: bool = Field(
        ..., description="Whether retrieval would help verify this claim"
    )
    suggested_query: str = Field(
        default="", description="Suggested search query for verification"
    )


class RetrievalOpportunity(BaseModel):
    """A specific opportunity for retrieval to enhance the text."""

    location: str = Field(..., description="Where in the text retrieval would help")
    reason: str = Field(..., description="Why retrieval is needed here")
    expected_benefit: str = Field(
        ..., description="How retrieval would improve the text"
    )
    priority: str = Field(default="medium", description="Priority: high, medium, low")


class SelfRAGResponse(BaseModel):
    """Response model following Self-RAG's reflection framework."""

    feedback: str = Field(
        ..., description="Overall reflection on content quality and information needs"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Specific suggestions for improving content with better information",
    )
    needs_improvement: bool = Field(
        ..., description="Whether the text needs retrieval or accuracy improvements"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence in the assessment"
    )

    # Self-RAG specific reflections
    overall_relevance: bool = Field(
        ...,
        description="ISREL: Is the overall content relevant to the intended purpose?",
    )
    overall_support: bool = Field(
        ..., description="ISSUP: Is the content well-supported by evidence?"
    )
    overall_utility: bool = Field(
        ..., description="ISUSE: Is the content useful for readers?"
    )

    # Detailed assessments
    factual_claims: list[FactualClaim] = Field(
        default_factory=list,
        description="Claims assessed with Self-RAG reflection criteria",
    )
    retrieval_opportunities: list[RetrievalOpportunity] = Field(
        default_factory=list,
        description="Specific opportunities where retrieval would enhance quality",
    )

    # Scoring
    relevance_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="How relevant is the content (ISREL score)",
    )
    support_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="How well-supported is the content (ISSUP score)",
    )
    utility_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="How useful is the content (ISUSE score)",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional Self-RAG reflection data"
    )


class SelfRAGCritic(BaseCritic):
    """Implements Self-RAG critique for factual accuracy.

    ## When to Use This Critic:

    ✅ When to use:
    - Fact-checking and verifying claims
    - Identifying unsupported assertions
    - Academic or research content requiring citations
    - Content where accuracy is paramount

    ❌ When to avoid:
    - Creative or fictional writing
    - Opinion pieces or personal narratives
    - When external verification isn't needed

    🎯 Best for:
    - Technical documentation
    - Research papers and reports
    - News articles and factual content
    - Educational materials requiring accuracy
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        # Initialize with custom config
        if config is None:
            config = Config()
        super().__init__(model, temperature, config, provider, api_key)

    @property
    def name(self) -> str:
        return "self_rag"

    def _get_response_type(self) -> type[BaseModel]:
        """Use custom SelfRAGResponse for structured output."""
        return SelfRAGResponse

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for RAG-style critique."""
        # Get previous context
        previous_context = self._get_previous_context(result)

        user_prompt = f"""You are a Self-RAG critic using reflection tokens to evaluate content quality.

Text to evaluate:
{text}
{previous_context}

Apply Self-RAG's reflection framework:

1. **Overall Assessment** (Reflection Tokens):
   - ISREL: Is the content relevant to its intended purpose?
   - ISSUP: Is the content supported by evidence or verifiable information?
   - ISUSE: Is the content useful and valuable for readers?

2. **Detailed Analysis**:
   - Identify all factual claims and assess each with ISREL/ISSUP/ISUSE
   - Find retrieval opportunities where external information would enhance quality
   - Evaluate the overall information sufficiency

3. **Adaptive Reflection**:
   - Where would retrieval most improve the content?
   - Which claims critically need verification?
   - How can the content better serve its purpose?

Focus on actionable improvements that would enhance relevance, support, and utility."""

        return [
            {
                "role": "system",
                "content": "You are a Self-RAG critic that uses reflection tokens (ISREL, ISSUP, ISUSE) to adaptively evaluate content and identify where retrieval would enhance quality. You reflect on relevance, support, and utility.",
            },
            {"role": "user", "content": user_prompt},
        ]
