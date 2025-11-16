"""Style critic implementation for voice, tone, and writing style transformation.

This module provides sophisticated style analysis and transformation capabilities,
enabling text to be adapted to match specific writing styles, brand voices, or
publication standards through detailed pattern recognition and application.

## Core Capabilities:

- **Style Analysis**: Deep pattern recognition across multiple style dimensions
- **Voice Transformation**: Active/passive voice adjustment and perspective shifts
- **Tone Adaptation**: Formal/informal, professional/casual tone calibration
- **Structural Matching**: Sentence patterns, rhythm, and flow replication
- **Vocabulary Alignment**: Word choice and complexity level matching

## Style Dimensions Analyzed:

1. **Tone**: Overall emotional and formality register
2. **Voice**: Active/passive balance, personal/impersonal perspective
3. **Sentence Structure**: Length, complexity, and variety patterns
4. **Vocabulary**: Word choice, technical level, and domain specificity
5. **Rhythm**: Pacing, punctuation patterns, and paragraph flow
6. **Distinctive Patterns**: Unique stylistic fingerprints and signatures

## Usage Patterns:

    >>> # Match a reference text style
    >>> critic = StyleCritic(
    ...     reference_text="Your brand voice sample text here...",
    ...     style_description="Professional yet approachable"
    ... )
    >>>
    >>> # Use specific style examples
    >>> critic = StyleCritic(
    ...     style_examples=[
    ...         "We're excited to announce...",
    ...         "Our team has discovered...",
    ...         "Join us in celebrating..."
    ...     ]
    ... )
    >>>
    >>> # Load style from file
    >>> critic = style_critic_from_file("brand_voice_guide.txt")

## Style Matching Process:

1. **Analysis Phase**: Extract patterns from reference material
2. **Comparison Phase**: Identify mismatches in current text
3. **Suggestion Phase**: Provide specific rewrites and adjustments
4. **Scoring Phase**: Quantify style alignment (0.0-1.0)

## Best Practices:

- Provide substantial reference text (500+ words) for accurate pattern extraction
- Combine reference text with explicit style descriptions for best results
- Use style examples for specific phrase patterns you want to maintain
- Review alignment scores: >0.8 = good match, <0.6 = significant work needed

This critic excels at maintaining brand consistency, adapting content for different
audiences, and ensuring stylistic coherence across document collections.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..core.config import Config
from ..core.llm_client import Provider
from ..core.models import SifakaResult
from .core.base import BaseCritic


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
        """Initialize the Style critic for voice and tone transformation.

        Creates a style-aware critic that analyzes and transforms text to match
        specific writing styles, brand voices, or publication standards. Supports
        multiple configuration approaches from reference text to explicit descriptions.

        Args:
            model: LLM model for style analysis. GPT-4o-mini provides good
                balance of quality and cost for style matching.
            temperature: Generation temperature (0.0-1.0). Higher values (0.7-0.9)
                allow more creative style adaptations, lower values (0.3-0.5)
                ensure more consistent matching.
            provider: LLM provider (OpenAI, Anthropic, etc.). Defaults to
                environment configuration.
            api_key: API key override if not using environment variables
            config: Full Sifaka configuration object. Style-specific fields
                (style_reference_text, style_description, style_examples)
                override individual parameters if present.
            reference_text: Sample text exemplifying the target style. Provide
                500+ words for best pattern extraction. This is the most
                effective way to capture complex style nuances.
            style_description: Explicit description of desired style characteristics
                (e.g., "Professional yet approachable with active voice").
                Combine with reference_text for best results.
            style_examples: List of specific phrases or sentences that exemplify
                the target style. Useful for capturing signature expressions
                or standard openings/closings.

        Example:
            >>> # Brand voice from reference document
            >>> critic = StyleCritic(
            ...     reference_text=open("brand_voice_guide.txt").read(),
            ...     style_description="Conversational but authoritative"
            ... )
            >>>
            >>> # Academic style with examples
            >>> critic = StyleCritic(
            ...     style_description="Academic writing with passive voice",
            ...     style_examples=[
            ...         "It has been demonstrated that...",
            ...         "The results indicate a significant correlation...",
            ...         "Further research is warranted to explore..."
            ...     ]
            ... )
            >>>
            >>> # Marketing copy style
            >>> critic = StyleCritic(
            ...     model="gpt-4",  # Higher quality for nuanced marketing copy
            ...     temperature=0.8,  # More creative adaptations
            ...     style_description="Persuasive, benefit-focused, action-oriented"
            ... )

        Best practices:
            - Provide substantial reference text (500+ words) when available
            - Combine reference_text with style_description for clarity
            - Use style_examples for specific phrases you want to preserve
            - Consider using GPT-4 for highly nuanced style matching
            - Test with representative content to tune temperature
        """
        super().__init__(model, temperature, config, provider, api_key)
        self.reference_text = reference_text
        self.style_description = style_description
        self.style_examples = style_examples or []

        # Allow config to override if provided
        if config:
            # Check if config has a critic attribute (it's the full Config object)
            critic_config = getattr(config, "critic", config)

            if (
                hasattr(critic_config, "style_reference_text")
                and critic_config.style_reference_text
            ):
                self.reference_text = critic_config.style_reference_text
            if (
                hasattr(critic_config, "style_description")
                and critic_config.style_description
            ):
                self.style_description = critic_config.style_description
            if (
                hasattr(critic_config, "style_examples")
                and critic_config.style_examples
            ):
                self.style_examples = critic_config.style_examples

    @property
    def name(self) -> str:
        """Return the unique identifier for this critic.

        Returns:
            "style" - Used in configuration, logging, and result metadata
            to identify this critic's contributions.
        """
        return "style"

    def _get_response_type(self) -> type[BaseModel]:
        """Specify the structured response format for style analysis.

        Returns:
            StyleResponse class that includes:
            - Detailed style analysis breakdown
            - Specific improvement suggestions
            - Style alignment score (0.0-1.0)
            - Target style characteristics

        Note:
            The structured response ensures consistent, actionable feedback
            with quantifiable style matching metrics.
        """
        return StyleResponse

    async def _create_messages(
        self, text: str, result: SifakaResult
    ) -> List[Dict[str, str]]:
        """Create detailed prompts for comprehensive style analysis.

        Constructs a sophisticated prompt that guides the LLM to perform
        deep style analysis, identify specific mismatches, and provide
        actionable transformation suggestions.

        Args:
            text: Text to analyze for style alignment. The critic examines
                sentence structure, vocabulary, tone, voice, and rhythm.
            result: SifakaResult containing iteration history and context.
                Previous feedback is incorporated to avoid repetition.

        Returns:
            List of message dictionaries with:
            - System message: Expert style analyst persona with focus on
              specificity and actionable feedback
            - User message: Structured analysis request with target style
              guidance and detailed evaluation criteria

        Analysis dimensions:
            1. **Sentence Structure**: Length, complexity, variety patterns
            2. **Voice**: Active/passive balance, perspective consistency
            3. **Tone**: Formality level, emotional register, professional markers
            4. **Vocabulary**: Word choice, technical level, domain specificity
            5. **Rhythm**: Pacing, punctuation patterns, paragraph flow

        The prompt emphasizes:
            - Specific examples over general observations
            - Exact rewrites rather than vague suggestions
            - Quantifiable metrics (sentence length, voice ratios)
            - Before/after comparisons for clarity
        """
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
        """Construct comprehensive style guidance from available sources.

        Assembles style requirements from multiple inputs (reference text,
        descriptions, examples) into a unified guidance section that the
        LLM uses for style matching.

        Returns:
            Formatted string containing:
            - Target style requirements (if description provided)
            - Reference text analysis instructions (if reference provided)
            - Specific style examples to incorporate (if examples provided)
            - Default professional style guidance (if no inputs provided)

        Priority order:
            1. Explicit style description (clearest intent)
            2. Reference text (richest pattern source)
            3. Style examples (specific patterns to match)
            4. Default guidelines (fallback for consistency)

        Text processing:
            - Reference texts over 2000 chars are truncated with ellipsis
            - Maximum 10 style examples are included to avoid prompt bloat
            - Each section includes specific extraction instructions

        Example output:
            ```
            TARGET STYLE REQUIREMENTS:
            Professional yet approachable, active voice preferred

            REFERENCE TEXT TO MATCH:
            [First 2000 chars of reference...]

            Key style elements to extract from this reference:
            - Typical sentence length and structure
            - Common phrases and word choices
            - Tone and formality level
            - Punctuation and formatting patterns
            - Voice (first/third person, active/passive)

            SPECIFIC STYLE EXAMPLES TO INCORPORATE:
            - "We're excited to announce..."
            - "Our team has discovered..."

            Use these exact phrases or similar patterns where appropriate.
            ```
        """
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
            sections.append(
                f"""REFERENCE TEXT TO MATCH:
{ref_text}

Key style elements to extract from this reference:
- Typical sentence length and structure
- Common phrases and word choices
- Tone and formality level
- Punctuation and formatting patterns
- Voice (first/third person, active/passive)"""
            )

        if self.style_examples:
            examples = "\n".join(
                f"- {ex}" for ex in self.style_examples[:10]
            )  # Limit to 10 examples
            sections.append(
                f"""SPECIFIC STYLE EXAMPLES TO INCORPORATE:
{examples}

Use these exact phrases or similar patterns where appropriate."""
            )

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
        """Adjust response metrics for accuracy and consistency.

        Post-processes the LLM's style analysis to ensure logical consistency
        between different response fields and adjust confidence based on
        available style guidance.

        Args:
            response: Raw StyleResponse from LLM containing initial analysis

        Returns:
            Adjusted StyleResponse with:
            - Confidence capped at 0.5 if no style guidance was provided
            - Alignment score synchronized with needs_improvement flag
            - Logical consistency between all metrics

        Adjustments made:
            1. **Confidence adjustment**: Without reference text or description,
               confidence is capped at 0.5 since style matching is guesswork
            2. **Alignment synchronization**:
               - If needs_improvement=True, alignment score ≤ 0.7
               - If needs_improvement=False, alignment score ≥ 0.7
            3. **Score normalization**: Ensures scores stay within valid ranges

        This ensures users receive consistent, interpretable metrics where:
            - Low confidence indicates limited style guidance
            - Alignment scores align with improvement recommendations
            - All metrics tell a coherent story about style matching
        """
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
    """Create a StyleCritic using a reference file for style matching.

    Convenience function that loads style reference text from a file and
    creates a configured StyleCritic. Ideal for maintaining consistent
    style across projects by using a standard reference document.

    Args:
        style_file_path: Path to file containing reference text that
            exemplifies the target style. Can be absolute or relative path.
            Supports any text file format (.txt, .md, .doc content as text).
        model: LLM model for style analysis. Defaults to gpt-4o-mini for
            cost-effective style matching. Consider gpt-4 for nuanced styles.
        temperature: Generation temperature (0.0-1.0). Default 0.7 balances
            consistency with creative adaptation.
        **kwargs: Additional arguments passed to StyleCritic.
            See CriticFactoryParams in core.type_defs for base fields.
            StyleCritic-specific additional fields:
            - style_description: Additional style clarification
            - style_examples: Specific phrases to incorporate

    Returns:
        Configured StyleCritic instance ready for style transformation

    Raises:
        ValueError: If the style file cannot be read, including:
            - File not found
            - Permission denied
            - Encoding errors

    Example:
        >>> # Load brand voice from file
        >>> critic = style_critic_from_file("brand_voice_guide.txt")
        >>>
        >>> # Academic style with additional description
        >>> critic = style_critic_from_file(
        ...     "academic_sample.txt",
        ...     model="gpt-4",
        ...     style_description="Formal academic tone with citations"
        ... )
        >>>
        >>> # Marketing style with high creativity
        >>> critic = style_critic_from_file(
        ...     "marketing_copy_samples.txt",
        ...     temperature=0.9
        ... )

    Best practices:
        - Use substantial reference files (500+ words) for best results
        - Include diverse examples of the target style in the file
        - Consider adding style_description for clarity
        - Test with representative content to validate matching
        - Store reference files in version control for consistency
    """
    try:
        with open(style_file_path, encoding="utf-8") as f:
            reference_text = f.read()
    except Exception as e:
        raise ValueError(f"Could not read style file: {e}")

    return StyleCritic(
        model=model, temperature=temperature, reference_text=reference_text, **kwargs
    )
