"""
Formatting and style rules for Sifaka.

This module provides rules for validating text formatting, including length,
paragraph structure, writing style, and general formatting patterns.
"""

from typing import Dict, Any, List, Protocol, runtime_checkable, Final, TypeVar, Set
from typing_extensions import TypeGuard
from dataclasses import dataclass, field
import re
from sifaka.rules.base import Rule, RuleResult, RuleConfig, RuleValidator


@dataclass(frozen=True)
class LengthConfig(RuleConfig):
    """Configuration for text length validation."""

    min_length: int = 50
    max_length: int = 5000
    exact_length: int | None = None
    unit: str = "characters"  # "characters" or "words"
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.min_length < 0:
            raise ValueError("min_length must be non-negative")
        if self.max_length < self.min_length:
            raise ValueError("max_length must be greater than or equal to min_length")
        if self.exact_length is not None and self.exact_length < 0:
            raise ValueError("exact_length must be non-negative")
        if self.unit not in ["characters", "words"]:
            raise ValueError("unit must be either 'characters' or 'words'")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True)
class ParagraphConfig(RuleConfig):
    """Configuration for paragraph structure validation."""

    min_sentences: int = 2
    max_sentences: int = 5
    min_words: int = 5
    max_words: int = 30
    sentence_delimiters: str = r"[.!?]+"
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.min_sentences < 0:
            raise ValueError("min_sentences must be non-negative")
        if self.max_sentences < self.min_sentences:
            raise ValueError("max_sentences must be greater than or equal to min_sentences")
        if self.min_words < 0:
            raise ValueError("min_words must be non-negative")
        if self.max_words < self.min_words:
            raise ValueError("max_words must be greater than or equal to min_words")
        if not self.sentence_delimiters:
            raise ValueError("sentence_delimiters cannot be empty")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True)
class StyleConfig(RuleConfig):
    """Configuration for writing style validation."""

    style_indicators: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "formal": ["therefore", "consequently", "furthermore", "thus", "hence"],
            "informal": ["yeah", "cool", "awesome", "btw", "gonna", "wanna"],
            "technical": ["algorithm", "parameter", "function", "variable", "method"],
            "casual": ["hey", "hi", "thanks", "please", "sorry"],
        }
    )
    style_threshold: float = 0.7
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.style_indicators, dict):
            raise ValueError("style_indicators must be a Dict[str, List[str]]")
        if not all(isinstance(v, list) for v in self.style_indicators.values()):
            raise ValueError("style_indicators values must be lists")
        if not 0.0 <= self.style_threshold <= 1.0:
            raise ValueError("style_threshold must be between 0.0 and 1.0")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True)
class FormattingConfig(RuleConfig):
    """Configuration for general text formatting validation."""

    formatting_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            "bullet_points": r"^\s*[-*•]\s+",
            "numbered_lists": r"^\s*\d+\.\s+",
            "indentation": r"^\s{2,}",
            "blank_lines": r"^\s*$",
        }
    )
    required_patterns: Set[str] = field(default_factory=set)
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.formatting_patterns, dict):
            raise ValueError("formatting_patterns must be a Dict[str, str]")
        if not all(isinstance(v, str) for v in self.formatting_patterns.values()):
            raise ValueError("formatting_patterns values must be strings")
        if not all(p in self.formatting_patterns for p in self.required_patterns):
            raise ValueError("required_patterns must be a subset of formatting_patterns keys")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@runtime_checkable
class LengthValidator(Protocol):
    """Protocol for text length validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> LengthConfig: ...


@runtime_checkable
class ParagraphValidator(Protocol):
    """Protocol for paragraph structure validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> ParagraphConfig: ...


@runtime_checkable
class StyleValidator(Protocol):
    """Protocol for writing style validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> StyleConfig: ...


@runtime_checkable
class FormattingValidator(Protocol):
    """Protocol for general text formatting validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> FormattingConfig: ...


class DefaultLengthValidator:
    """Default implementation of text length validation."""

    def __init__(self, config: LengthConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> LengthConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text length."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        length = len(text) if self.config.unit == "characters" else len(text.split())

        if self.config.exact_length is not None and length != self.config.exact_length:
            return RuleResult(
                passed=False,
                message=f"Text length {length} does not meet exact length requirement of {self.config.exact_length} {self.config.unit}",
                metadata={"length": length, "requirement": "exact"},
            )

        if length < self.config.min_length:
            return RuleResult(
                passed=False,
                message=f"Text length {length} is below minimum {self.config.min_length} {self.config.unit}",
                metadata={"length": length, "requirement": "minimum"},
            )

        if length > self.config.max_length:
            return RuleResult(
                passed=False,
                message=f"Text length {length} exceeds maximum {self.config.max_length} {self.config.unit}",
                metadata={"length": length, "requirement": "maximum"},
            )

        return RuleResult(
            passed=True,
            message=f"Text length {length} meets requirements",
            metadata={"length": length},
        )


class DefaultParagraphValidator:
    """Default implementation of paragraph structure validation."""

    def __init__(self, config: ParagraphConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> ParagraphConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate paragraph structure."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        issues = []
        metadata = {"paragraphs": []}

        for i, paragraph in enumerate(paragraphs, 1):
            sentences = [
                s.strip() for s in re.split(self.config.sentence_delimiters, paragraph) if s.strip()
            ]
            words = paragraph.split()

            paragraph_info = {
                "num_sentences": len(sentences),
                "num_words": len(words),
                "content": paragraph[:100] + "..." if len(paragraph) > 100 else paragraph,
            }

            if len(sentences) < self.config.min_sentences:
                issues.append(f"Paragraph {i} has fewer than {self.config.min_sentences} sentences")
                paragraph_info["error"] = "too_few_sentences"
            elif len(sentences) > self.config.max_sentences:
                issues.append(f"Paragraph {i} exceeds {self.config.max_sentences} sentences")
                paragraph_info["error"] = "too_many_sentences"

            if len(words) < self.config.min_words:
                issues.append(f"Paragraph {i} has fewer than {self.config.min_words} words")
                paragraph_info["error"] = "too_few_words"
            elif len(words) > self.config.max_words:
                issues.append(f"Paragraph {i} exceeds {self.config.max_words} words")
                paragraph_info["error"] = "too_many_words"

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


class DefaultStyleValidator:
    """Default implementation of writing style validation."""

    def __init__(self, config: StyleConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> StyleConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate writing style consistency."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        text_lower = text.lower()
        style_scores = {}

        for style, indicators in self.config.style_indicators.items():
            found_indicators = [ind for ind in indicators if ind in text_lower]
            style_scores[style] = len(found_indicators) / len(indicators)

        # Find the dominant style
        dominant_style, max_score = max(style_scores.items(), key=lambda x: x[1])

        # Check if the dominant style is sufficiently strong
        if max_score < self.config.style_threshold:
            return RuleResult(
                passed=False,
                message="No clear writing style detected",
                metadata={
                    "style_scores": style_scores,
                    "threshold": self.config.style_threshold,
                },
            )

        # Check for style consistency
        inconsistent_styles = [
            style
            for style, score in style_scores.items()
            if style != dominant_style and score > self.config.style_threshold / 2
        ]

        if inconsistent_styles:
            return RuleResult(
                passed=False,
                message=f"Mixed writing styles detected: {dominant_style} with {', '.join(inconsistent_styles)}",
                metadata={
                    "dominant_style": dominant_style,
                    "inconsistent_styles": inconsistent_styles,
                    "style_scores": style_scores,
                },
            )

        return RuleResult(
            passed=True,
            message=f"Consistent {dominant_style} writing style detected",
            metadata={
                "dominant_style": dominant_style,
                "style_scores": style_scores,
            },
        )


class DefaultFormattingValidator:
    """Default implementation of text formatting validation."""

    def __init__(self, config: FormattingConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> FormattingConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text formatting patterns."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        lines = text.split("\n")
        found_patterns = set()
        pattern_matches = {}

        for pattern_name, pattern in self.config.formatting_patterns.items():
            matches = [i + 1 for i, line in enumerate(lines) if re.match(pattern, line)]
            if matches:
                found_patterns.add(pattern_name)
                pattern_matches[pattern_name] = matches

        missing_patterns = self.config.required_patterns - found_patterns
        if missing_patterns:
            return RuleResult(
                passed=False,
                message=f"Missing required formatting patterns: {', '.join(missing_patterns)}",
                metadata={
                    "missing_patterns": list(missing_patterns),
                    "found_patterns": list(found_patterns),
                    "pattern_matches": pattern_matches,
                },
            )

        return RuleResult(
            passed=True,
            message="All required formatting patterns found",
            metadata={
                "found_patterns": list(found_patterns),
                "pattern_matches": pattern_matches,
            },
        )


class LengthRule(Rule):
    """Rule that checks for appropriate text length."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: LengthValidator,
    ) -> None:
        """Initialize the length rule."""
        super().__init__(name=name, description=description)
        self._validator = validator

    @property
    def validator(self) -> LengthValidator:
        """Get the length validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """Validate text length."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            return self._validator.validate(text)
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during length validation: {str(e)}",
                metadata={"error": str(e)},
            )


class ParagraphRule(Rule):
    """Rule that checks for proper paragraph formatting."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: ParagraphValidator,
    ) -> None:
        """Initialize the paragraph rule."""
        super().__init__(name=name, description=description)
        self._validator = validator

    @property
    def validator(self) -> ParagraphValidator:
        """Get the paragraph validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """Validate paragraph structure."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            return self._validator.validate(text)
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during paragraph validation: {str(e)}",
                metadata={"error": str(e)},
            )


class StyleRule(Rule):
    """Rule that checks for consistent writing style."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: StyleValidator,
    ) -> None:
        """Initialize the style rule."""
        super().__init__(name=name, description=description)
        self._validator = validator

    @property
    def validator(self) -> StyleValidator:
        """Get the style validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """Validate writing style."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            return self._validator.validate(text)
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during style validation: {str(e)}",
                metadata={"error": str(e)},
            )


class FormattingRule(Rule):
    """Rule that checks for proper text formatting."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: FormattingValidator,
    ) -> None:
        """Initialize the formatting rule."""
        super().__init__(name=name, description=description)
        self._validator = validator

    @property
    def validator(self) -> FormattingValidator:
        """Get the formatting validator."""
        return self._validator

    def validate(self, text: str) -> RuleResult:
        """Validate text formatting."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        try:
            return self._validator.validate(text)
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during formatting validation: {str(e)}",
                metadata={"error": str(e)},
            )


def create_length_rule(
    name: str,
    description: str,
    config: LengthConfig | None = None,
) -> LengthRule:
    """Create a length rule with default configuration."""
    validator = DefaultLengthValidator(config or LengthConfig())
    return LengthRule(name=name, description=description, validator=validator)


def create_paragraph_rule(
    name: str,
    description: str,
    config: ParagraphConfig | None = None,
) -> ParagraphRule:
    """Create a paragraph rule with default configuration."""
    validator = DefaultParagraphValidator(config or ParagraphConfig())
    return ParagraphRule(name=name, description=description, validator=validator)


def create_style_rule(
    name: str,
    description: str,
    config: StyleConfig | None = None,
) -> StyleRule:
    """Create a style rule with default configuration."""
    validator = DefaultStyleValidator(config or StyleConfig())
    return StyleRule(name=name, description=description, validator=validator)


def create_formatting_rule(
    name: str,
    description: str,
    config: FormattingConfig | None = None,
) -> FormattingRule:
    """Create a formatting rule with default configuration."""
    validator = DefaultFormattingValidator(config or FormattingConfig())
    return FormattingRule(name=name, description=description, validator=validator)
