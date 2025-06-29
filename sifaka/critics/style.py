"""Style critic implementation.

The Style critic helps transform text to match a specific writing style,
voice, or tone by analyzing reference text and applying its characteristics.

This critic is useful for:
- Matching brand voice and tone
- Adapting content to specific audiences
- Maintaining consistent writing style across documents
- Emulating author styles or publication standards
"""

from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field

from ..core.models import SifakaResult
from ..core.llm_client import Provider
from .core.base import BaseCritic
from ..core.config import Config


class StyleAnalysis(BaseModel):
    """Analysis of style characteristics."""

    tone: str = Field(..., description="Overall tone (formal, casual, academic, etc.)")
    voice: str = Field(
        ..., description="Voice characteristics (active/passive, personal/impersonal)"
    )
    sentence_structure: str = Field(..., description="Sentence patterns and complexity")
    vocabulary: str = Field(..., description="Word choice and complexity level")
    rhythm: str = Field(..., description="Pacing and flow characteristics")
    key_patterns: List[str] = Field(
        default_factory=list, description="Distinctive style patterns"
    )


class StyleResponse(BaseModel):
    """Response model for style critique."""

    feedback: str = Field(..., description="Detailed feedback on style alignment")
    suggestions: List[str] = Field(
        default_factory=list, description="Specific suggestions for style improvement"
    )
    needs_improvement: bool = Field(
        ..., description="Whether the text needs style adjustments"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence in the assessment"
    )
    style_analysis: Optional[StyleAnalysis] = Field(
        default=None, description="Analysis of target style characteristics"
    )
    alignment_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How well the text matches target style",
    )


class StyleCritic(BaseCritic):
    """Analyzes and improves text to match a specific style.

    ## When to Use This Critic:

    ✅ When to use:
    - Matching brand voice guidelines
    - Adapting content for different audiences
    - Maintaining consistent style across documents
    - Emulating specific author or publication styles

    ❌ When to avoid:
    - Technical documentation requiring precision
    - Legal or regulatory content
    - When originality is more important than consistency

    🎯 Best for:
    - Marketing and brand content
    - Blog posts and articles
    - Creative writing adaptations
    - Email and communication templates
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        provider: Optional[Union[str, Provider]] = None,
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
        reference_text: Optional[str] = None,
        style_description: Optional[str] = None,
        style_examples: Optional[List[str]] = None,
    ):
        """Initialize the Style critic.

        Args:
            model: LLM model to use
            temperature: Generation temperature
            provider: LLM provider
            api_key: API key override
            config: Full Sifaka configuration
            reference_text: Text exemplifying the target style
            style_description: Description of the desired style
            style_examples: List of example phrases/sentences in target style
        """
        super().__init__(model, temperature, config, provider, api_key)
        self.reference_text = reference_text
        self.style_description = style_description
        self.style_examples = style_examples or []

        # Allow config to override if provided
        if config:
            if hasattr(config, "style_reference_text") and config.style_reference_text:
                self.reference_text = config.style_reference_text
            if hasattr(config, "style_description") and config.style_description:
                self.style_description = config.style_description
            if hasattr(config, "style_examples") and config.style_examples:
                self.style_examples = config.style_examples

    @property
    def name(self) -> str:
        return "style"

    def _get_response_type(self) -> type[BaseModel]:
        """Use custom StyleResponse for structured output."""
        return StyleResponse

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create messages for style critique."""
        # Get previous context
        previous_context = self._get_previous_context(result)

        # Build style guidance
        style_guidance = self._build_style_guidance()

        user_prompt = f"""Analyze this text's style compared to the target style.

TEXT TO ANALYZE:
{text}
{previous_context}

{style_guidance}

PROVIDE SPECIFIC ANALYSIS:

1. STYLE PATTERNS FOUND:
   - Sentence structure (simple/complex, length variation)
   - Voice (active/passive ratio)
   - Tone markers (formal/informal words)
   - Punctuation style
   
2. SPECIFIC MISMATCHES:
   - Quote exact phrases that don't match the target style
   - Explain WHY each doesn't match
   - Show how the target style would handle it

3. ACTIONABLE CHANGES:
   - List 3-5 specific rewrites with before/after examples
   - Focus on the most impactful changes first
   - Be precise: "Change X to Y" not "consider changing"

4. ALIGNMENT SCORE: X/10 (convert to 0.0-1.0 scale)
   - Vocabulary match (0.3 weight)
   - Sentence structure match (0.3 weight)  
   - Tone/voice match (0.4 weight)

Set needs_improvement=true if alignment < 0.8

Be specific and actionable. Show, don't just tell."""

        return [
            {
                "role": "system",
                "content": """You are an expert writing style analyst and editor. Your role is to:

1. Identify SPECIFIC style patterns (not just "formal" or "casual")
2. Give CONCRETE examples from the text showing style mismatches
3. Provide EXACT rewriting suggestions, not vague advice
4. Focus on measurable aspects: sentence length, word choice, punctuation, voice
5. Ignore content/accuracy - ONLY evaluate writing style

When analyzing, be extremely specific. Instead of "make it more formal", say "replace 'got' with 'obtained', remove contractions like 'don't' → 'do not', increase average sentence length from 12 to 18 words".""",
            },
            {"role": "user", "content": user_prompt},
        ]

    def _build_style_guidance(self) -> str:
        """Build the style guidance section based on available inputs."""
        sections = []

        if self.style_description:
            sections.append(f"TARGET STYLE REQUIREMENTS:\n{self.style_description}")

        if self.reference_text:
            # Truncate if too long
            ref_text = (
                self.reference_text[:2000] + "..."
                if len(self.reference_text) > 2000
                else self.reference_text
            )
            sections.append(f"""REFERENCE TEXT TO MATCH:
{ref_text}

Key style elements to extract from this reference:
- Typical sentence length and structure
- Common phrases and word choices  
- Tone and formality level
- Punctuation and formatting patterns
- Voice (first/third person, active/passive)""")

        if self.style_examples:
            examples = "\n".join(
                f"- {ex}" for ex in self.style_examples[:10]
            )  # Limit to 10 examples
            sections.append(f"""SPECIFIC STYLE EXAMPLES TO INCORPORATE:
{examples}

Use these exact phrases or similar patterns where appropriate.""")

        if not sections:
            # More specific default guidance
            sections.append(
                """TARGET STYLE: Clear, professional writing with:
- Active voice (>80% of sentences)
- Average sentence length: 15-20 words
- Minimal jargon
- Consistent present tense
- Direct, declarative statements"""
            )

        return "\n\n".join(sections)

    def _post_process_response(self, response: StyleResponse) -> StyleResponse:
        """Post-process the response to ensure quality."""
        # Adjust confidence based on how much style guidance was provided
        if not self.reference_text and not self.style_description:
            response.confidence = min(response.confidence, 0.5)

        # Ensure alignment score is reasonable
        if response.needs_improvement and response.alignment_score > 0.8:
            response.alignment_score = min(response.alignment_score, 0.7)
        elif not response.needs_improvement and response.alignment_score < 0.7:
            response.alignment_score = max(response.alignment_score, 0.7)

        return response


# Convenience function for creating a style critic from a file
def style_critic_from_file(
    style_file_path: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    **kwargs: Any,
) -> StyleCritic:
    """Create a StyleCritic using a reference file.

    Args:
        style_file_path: Path to file containing reference text
        model: LLM model to use
        temperature: Generation temperature
        **kwargs: Additional arguments for StyleCritic

    Returns:
        Configured StyleCritic instance
    """
    try:
        with open(style_file_path, "r", encoding="utf-8") as f:
            reference_text = f.read()
    except Exception as e:
        raise ValueError(f"Could not read style file: {e}")

    return StyleCritic(
        model=model, temperature=temperature, reference_text=reference_text, **kwargs
    )
