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
    """

    min_length: int = Field(default=50)
    max_length: int = Field(default=5000)

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
            ValueError: If output is None
        """
        if output is None:
            raise ValueError("Output cannot be None")

        length = len(output)

        if length < self.min_length:
            return RuleResult(
                passed=False,
                message=f"Output is too short ({length} characters)",
                metadata={"length": length, "min_length": self.min_length},
            )

        if length > self.max_length:
            return RuleResult(
                passed=False,
                message=f"Output is too long ({length} characters)",
                metadata={"length": length, "max_length": self.max_length},
            )

        return RuleResult(
            passed=True,
            message=f"Output length is acceptable ({length} characters)",
            metadata={"length": length},
        )


class ParagraphRule(Rule):
    """
    Rule that checks for proper paragraph formatting.

    Attributes:
        min_sentences (int): Minimum sentences per paragraph
        max_sentences (int): Maximum sentences per paragraph
        min_words (int): Minimum words per sentence
        max_words (int): Maximum words per sentence
    """

    min_sentences: int = Field(default=2)
    max_sentences: int = Field(default=5)
    min_words: int = Field(default=5)
    max_words: int = Field(default=30)

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output has proper paragraph structure.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If output is None
        """
        if output is None:
            raise ValueError("Output cannot be None")

        paragraphs = output.split("\n\n")
        issues = []

        for i, paragraph in enumerate(paragraphs):
            sentences = re.split(r"[.!?]+", paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) < self.min_sentences:
                issues.append(f"Paragraph {i+1} has too few sentences ({len(sentences)})")

            if len(sentences) > self.max_sentences:
                issues.append(f"Paragraph {i+1} has too many sentences ({len(sentences)})")

            for j, sentence in enumerate(sentences):
                words = sentence.split()
                if len(words) < self.min_words:
                    issues.append(
                        f"Sentence {j+1} in paragraph {i+1} has too few words ({len(words)})"
                    )
                if len(words) > self.max_words:
                    issues.append(
                        f"Sentence {j+1} in paragraph {i+1} has too many words ({len(words)})"
                    )

        if issues:
            return RuleResult(
                passed=False,
                message="Paragraph formatting issues detected",
                metadata={"issues": issues},
            )

        return RuleResult(passed=True, message="Paragraph formatting is acceptable")


class StyleRule(Rule):
    """
    Rule that checks for consistent writing style.

    Attributes:
        style_indicators (Dict[str, List[str]]): Dictionary of style indicators
        style_threshold (float): Threshold for style consistency (0.0 to 1.0)
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

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output maintains a consistent writing style.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If output is None
        """
        if output is None:
            raise ValueError("Output cannot be None")

        output_lower = output.lower()
        style_scores = {}

        for style, indicators in self.style_indicators.items():
            found_indicators = []
            for indicator in indicators:
                if indicator in output_lower:
                    found_indicators.append(indicator)
            style_scores[style] = len(found_indicators) / len(indicators)

        # Find the dominant style
        dominant_style = max(style_scores.items(), key=lambda x: x[1])

        if dominant_style[1] < self.style_threshold:
            return RuleResult(
                passed=False,
                message="Writing style is inconsistent",
                metadata={"style_scores": style_scores},
            )

        return RuleResult(
            passed=True,
            message=f"Writing style is consistent ({dominant_style[0]})",
            metadata={"style_scores": style_scores},
        )


class FormattingRule(Rule):
    """
    Rule that checks for proper text formatting.

    Attributes:
        formatting_patterns (Dict[str, str]): Dictionary of formatting patterns to check
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

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output follows proper formatting rules.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If output is None
        """
        if output is None:
            raise ValueError("Output cannot be None")

        issues = {}

        for issue_type, pattern in self.formatting_patterns.items():
            matches = re.findall(pattern, output)
            if matches:
                issues[issue_type] = matches

        if issues:
            return RuleResult(
                passed=False, message="Formatting issues detected", metadata={"issues": issues}
            )

        return RuleResult(passed=True, message="Formatting is acceptable")
