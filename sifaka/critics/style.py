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

        user_prompt = f"""Analyze this text and provide feedback on how well it matches the target style:

Text to analyze:
{text}
{previous_context}

{style_guidance}

Please:
1. Analyze how well the text matches the target style
2. Identify specific areas where the style differs
3. Provide actionable suggestions for style alignment
4. Consider tone, voice, sentence structure, vocabulary, and rhythm
5. Give an alignment score (0.0 = completely different, 1.0 = perfect match)

Focus on style and voice rather than content accuracy or completeness."""

        return [
            {
                "role": "system",
                "content": "You are a style expert who helps writers match specific writing styles, voices, and tones. You analyze text for style characteristics and provide specific, actionable feedback on achieving the target style.",
            },
            {"role": "user", "content": user_prompt},
        ]

    def _build_style_guidance(self) -> str:
        """Build the style guidance section based on available inputs."""
        sections = []

        if self.style_description:
            sections.append(f"Target Style Description:\n{self.style_description}")

        if self.reference_text:
            # Truncate if too long
            ref_text = (
                self.reference_text[:2000] + "..."
                if len(self.reference_text) > 2000
                else self.reference_text
            )
            sections.append(f"Reference Text (exemplifying target style):\n{ref_text}")

        if self.style_examples:
            examples = "\n".join(
                f"- {ex}" for ex in self.style_examples[:10]
            )  # Limit to 10 examples
            sections.append(f"Style Examples:\n{examples}")

        if not sections:
            # Default guidance if nothing specific provided
            sections.append(
                """Target Style: Professional and clear writing that:
- Uses active voice and concrete language
- Maintains consistent tone throughout
- Balances clarity with engagement
- Follows standard style conventions"""
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
