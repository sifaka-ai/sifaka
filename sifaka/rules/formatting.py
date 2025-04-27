"""
Formatting and style rules for Sifaka.
"""

from typing import Dict, Any, List, Optional
from pydantic import Field
from sifaka.rules.base import Rule, RuleResult
import re


class LengthRule(Rule):
    """
    Rule that checks for appropriate text length.

    Attributes:
        min_length (int): Minimum allowed length in characters
        max_length (int): Maximum allowed length in characters
        exact_length (int): Exact allowed length in characters
        unit (str): Unit of measurement (characters or words)
    """

    min_length: int = Field(default=50)
    max_length: int = Field(default=5000)
    exact_length: int = Field(default=None)
    unit: str = Field(default="characters")

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output length is within acceptable bounds.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If output is None or not a string
        """
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        if output is None:
            raise ValueError("Output cannot be None")

        length = len(output)
        if self.unit == "words":
            length = len(output.split())

        if self.exact_length is not None and length != self.exact_length:
            return RuleResult(
                passed=False,
                message=f"Text length {length} does not meet exact length requirement of {self.exact_length} {self.unit}",
                metadata={"length": length},
            )

        if length < self.min_length:
            return RuleResult(
                passed=False,
                message=f"Text length {length} is below minimum {self.min_length} {self.unit}",
                metadata={"length": length},
            )

        if length > self.max_length:
            return RuleResult(
                passed=False,
                message=f"Text length {length} exceeds maximum {self.max_length} {self.unit}",
                metadata={"length": length},
            )

        return RuleResult(
            passed=True,
            message=f"Text length {length} meets requirements",
            metadata={"length": length},
        )


class ParagraphRule(Rule):
    """
    Rule that checks for proper paragraph formatting.

    Attributes:
        min_sentences: Minimum sentences per paragraph
        max_sentences: Maximum sentences per paragraph
        min_words: Minimum words per sentence
        max_words: Maximum words per sentence
    """

    min_sentences: int = Field(default=2)
    max_sentences: int = Field(default=5)
    min_words: int = Field(default=5)
    max_words: int = Field(default=30)

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output has proper paragraph structure.

        Args:
            output: The text to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results
        """
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        paragraphs = [p.strip() for p in output.split("\n\n") if p.strip()]
        issues = []
        metadata = {"paragraphs": []}

        for i, paragraph in enumerate(paragraphs, 1):
            sentences = [s.strip() for s in re.split(r"[.!?]+", paragraph) if s.strip()]
            words = paragraph.split()

            paragraph_info = {
                "num_sentences": len(sentences),
                "num_words": len(words),
                "content": paragraph,
            }

            if len(sentences) < self.min_sentences:
                issues.append(f"Paragraph {i} has fewer than {self.min_sentences} sentences")
                paragraph_info["error"] = "too few sentences"
            elif len(sentences) > self.max_sentences:
                issues.append(f"Paragraph {i} exceeds {self.max_sentences} sentences")
                paragraph_info["error"] = "too many sentences"

            if len(words) < self.min_words:
                issues.append(f"Paragraph {i} has fewer than {self.min_words} words")
                paragraph_info["error"] = "too few words"
            elif len(words) > self.max_words:
                issues.append(f"Paragraph {i} exceeds {self.max_words} words")
                paragraph_info["error"] = "too many words"

            metadata["paragraphs"].append(paragraph_info)

        if issues:
            return RuleResult(
                passed=False,
                message="Paragraph structure validation failed",
                metadata={"issues": issues, **metadata},
            )

        return RuleResult(
            passed=True,
            message="Paragraph structure is valid",
            metadata=metadata,
        )


class StyleRule(Rule):
    """
    Rule that checks for consistent writing style.

    Attributes:
        style_indicators: Dictionary of style indicators
        style_threshold: Threshold for style consistency (0.0 to 1.0)
    """

    style_indicators: Dict[str, List[str]] = Field(
        default={
            "formal": ["therefore", "consequently", "furthermore", "thus", "hence"],
            "informal": ["yeah", "cool", "awesome", "btw", "gonna", "wanna"],
            "technical": ["algorithm", "parameter", "function", "variable", "method"],
            "casual": ["hey", "hi", "thanks", "please", "sorry"],
        }
    )
    style_threshold: float = Field(default=0.7)

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output maintains a consistent writing style.

        Args:
            output: The text to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results
        """
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        style_scores = {}

        for style, indicators in self.style_indicators.items():
            found_indicators = [ind for ind in indicators if ind in output_lower]
            style_scores[style] = len(found_indicators) / len(indicators)

        # Find the dominant style
        dominant_style = max(style_scores.items(), key=lambda x: x[1])

        metadata = {
            "style_scores": style_scores,
            "dominant_style": dominant_style[0],
            "dominant_score": dominant_style[1],
        }

        if dominant_style[1] < self.style_threshold:
            return RuleResult(
                passed=False,
                message=f"Writing style is inconsistent (highest score: {dominant_style[1]:.2f})",
                metadata=metadata,
            )

        return RuleResult(
            passed=True,
            message=f"Writing style is consistent ({dominant_style[0]})",
            metadata=metadata,
        )


class FormattingRule(Rule):
    """
    Rule that checks for proper text formatting.

    Attributes:
        formatting_patterns: Dictionary of formatting patterns to check
    """

    formatting_patterns: Dict[str, str] = Field(
        default={
            "multiple_spaces": r"\s{2,}",
            "multiple_newlines": r"\n{3,}",
            "trailing_whitespace": r"\s+\n",
            "missing_period": r"[^.!?]\n",
            "missing_space": r"[a-z][A-Z]",
            "incorrect_quotes": r'["\'][^"\']*["\']',
        }
    )

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output follows proper formatting rules.

        Args:
            output: The text to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results
        """
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        issues = []
        metadata = {"issues": {}}

        for issue_type, pattern in self.formatting_patterns.items():
            matches = list(re.finditer(pattern, output))
            if matches:
                metadata["issues"][issue_type] = [m.group() for m in matches]
                issues.append(f"Found {issue_type} issues")

        if issues:
            return RuleResult(
                passed=False,
                message="Formatting issues detected",
                metadata=metadata,
            )

        return RuleResult(
            passed=True,
            message="Formatting is acceptable",
            metadata=metadata,
        )
