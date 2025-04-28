"""
Formatting and style rules for Sifaka.

This module provides rules for validating text formatting, including length,
paragraph structure, writing style, and general formatting patterns.
"""

import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

from sifaka.rules.base import Rule, RuleConfig, RuleResult, RuleValidator

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

class DefaultLengthValidator(RuleValidator[str]):
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
                message=f"Text {self.config.unit} count {length} does not match required count of {self.config.exact_length}",
                metadata={
                    "length": length,
                    "required_length": self.config.exact_length,
                    "unit": self.config.unit,
                },
            )

        if length < self.config.min_length:
            return RuleResult(
                passed=False,
                message=f"Text {self.config.unit} count {length} is below minimum of {self.config.min_length}",
                metadata={
                    "length": length,
                    "min_length": self.config.min_length,
                    "unit": self.config.unit,
                },
            )

        if length > self.config.max_length:
            return RuleResult(
                passed=False,
                message=f"Text {self.config.unit} count {length} exceeds maximum of {self.config.max_length}",
                metadata={
                    "length": length,
                    "max_length": self.config.max_length,
                    "unit": self.config.unit,
                },
            )

        return RuleResult(
            passed=True,
            message=f"Text {self.config.unit} count {length} is within acceptable range",
            metadata={
                "length": length,
                "min_length": self.config.min_length,
                "max_length": self.config.max_length,
                "unit": self.config.unit,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str

class DefaultParagraphValidator(RuleValidator[str]):
    """Default implementation of paragraph structure validation."""

    def __init__(self, config: ParagraphConfig) -> None:
        """Initialize with configuration."""
        self._config = config
        self._sentence_pattern = re.compile(self.config.sentence_delimiters)

    @property
    def config(self) -> ParagraphConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate paragraph structure."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        paragraphs = text.split("\n\n")
        invalid_paragraphs = []

        for i, para in enumerate(paragraphs, 1):
            # Count sentences
            sentences = [s.strip() for s in self._sentence_pattern.split(para) if s.strip()]
            sentence_count = len(sentences)

            # Count words in each sentence
            sentence_word_counts = [len(s.split()) for s in sentences]

            # Check sentence count
            if not (self.config.min_sentences <= sentence_count <= self.config.max_sentences):
                invalid_paragraphs.append(
                    {
                        "index": i,
                        "reason": "sentence_count",
                        "count": sentence_count,
                        "text": para,
                    }
                )
                continue

            # Check word count in each sentence
            for j, word_count in enumerate(sentence_word_counts, 1):
                if not (self.config.min_words <= word_count <= self.config.max_words):
                    invalid_paragraphs.append(
                        {
                            "index": i,
                            "sentence_index": j,
                            "reason": "word_count",
                            "count": word_count,
                            "text": sentences[j - 1],
                        }
                    )

        if invalid_paragraphs:
            return RuleResult(
                passed=False,
                message="Found paragraphs with invalid structure",
                metadata={
                    "invalid_paragraphs": invalid_paragraphs,
                    "total_paragraphs": len(paragraphs),
                },
            )

        return RuleResult(
            passed=True,
            message="All paragraphs have valid structure",
            metadata={
                "total_paragraphs": len(paragraphs),
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str

class DefaultStyleValidator(RuleValidator[str]):
    """Default implementation of writing style validation."""

    def __init__(self, config: StyleConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> StyleConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate writing style."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        text_lower = text.lower()
        style_scores = {}

        for style, indicators in self.config.style_indicators.items():
            matches = sum(1 for indicator in indicators if indicator.lower() in text_lower)
            if indicators:
                style_scores[style] = matches / len(indicators)

        if not style_scores:
            return RuleResult(
                passed=True,
                message="No style indicators found",
                metadata={"style_scores": {}},
            )

        # Find dominant style
        dominant_style = max(style_scores.items(), key=lambda x: x[1])
        is_consistent = dominant_style[1] >= self.config.style_threshold

        return RuleResult(
            passed=is_consistent,
            message=(
                f"Consistent {dominant_style[0]} style detected"
                if is_consistent
                else "No consistent writing style detected"
            ),
            metadata={
                "style_scores": style_scores,
                "dominant_style": dominant_style[0],
                "dominant_score": dominant_style[1],
                "threshold": self.config.style_threshold,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str

class DefaultFormattingValidator(RuleValidator[str]):
    """Default implementation of text formatting validation."""

    def __init__(self, config: FormattingConfig) -> None:
        """Initialize with configuration."""
        self._config = config
        self._compiled_patterns = {
            name: re.compile(pattern) for name, pattern in config.formatting_patterns.items()
        }

    @property
    def config(self) -> FormattingConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text formatting."""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        lines = text.split("\n")
        found_patterns = set()
        pattern_matches = {name: [] for name in self._compiled_patterns}

        for i, line in enumerate(lines, 1):
            for name, pattern in self._compiled_patterns.items():
                if pattern.search(line):
                    found_patterns.add(name)
                    pattern_matches[name].append(i)

        missing_required = self.config.required_patterns - found_patterns
        if missing_required:
            return RuleResult(
                passed=False,
                message=f"Missing required formatting patterns: {', '.join(missing_required)}",
                metadata={
                    "found_patterns": list(found_patterns),
                    "missing_patterns": list(missing_required),
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

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str

class LengthRule(Rule):
    """Rule for validating text length."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the rule with length validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        length_config = LengthConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultLengthValidator(length_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output length."""
        return self._validator.validate(output)

class ParagraphRule(Rule):
    """Rule for validating paragraph structure."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the rule with paragraph validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        paragraph_config = ParagraphConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultParagraphValidator(paragraph_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output paragraph structure."""
        return self._validator.validate(output)

class StyleRule(Rule):
    """Rule for validating writing style."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the rule with style validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        style_config = StyleConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultStyleValidator(style_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output writing style."""
        return self._validator.validate(output)

class FormattingRule(Rule):
    """Rule for validating text formatting."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the rule with formatting validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        formatting_config = FormattingConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultFormattingValidator(formatting_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output formatting."""
        return self._validator.validate(output)

def create_length_rule(
    name: str = "length_rule",
    description: str = "Validates text length",
    config: Optional[Dict[str, Any]] = None,
) -> LengthRule:
    """
    Create a length rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured LengthRule instance
    """
    if config is None:
        config = {
            "min_length": 50,
            "max_length": 5000,
            "exact_length": None,
            "unit": "characters",
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return LengthRule(
        name=name,
        description=description,
        config=config,
    )

def create_paragraph_rule(
    name: str = "paragraph_rule",
    description: str = "Validates paragraph structure",
    config: Optional[Dict[str, Any]] = None,
) -> ParagraphRule:
    """
    Create a paragraph rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured ParagraphRule instance
    """
    if config is None:
        config = {
            "min_sentences": 2,
            "max_sentences": 5,
            "min_words": 5,
            "max_words": 30,
            "sentence_delimiters": r"[.!?]+",
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return ParagraphRule(
        name=name,
        description=description,
        config=config,
    )

def create_style_rule(
    name: str = "style_rule",
    description: str = "Validates writing style",
    config: Optional[Dict[str, Any]] = None,
) -> StyleRule:
    """
    Create a style rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured StyleRule instance
    """
    if config is None:
        config = {
            "style_indicators": {
                "formal": ["therefore", "consequently", "furthermore", "thus", "hence"],
                "informal": ["yeah", "cool", "awesome", "btw", "gonna", "wanna"],
                "technical": ["algorithm", "parameter", "function", "variable", "method"],
                "casual": ["hey", "hi", "thanks", "please", "sorry"],
            },
            "style_threshold": 0.7,
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return StyleRule(
        name=name,
        description=description,
        config=config,
    )

def create_formatting_rule(
    name: str = "formatting_rule",
    description: str = "Validates text formatting",
    config: Optional[Dict[str, Any]] = None,
) -> FormattingRule:
    """
    Create a formatting rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured FormattingRule instance
    """
    if config is None:
        config = {
            "formatting_patterns": {
                "bullet_points": r"^\s*[-*•]\s+",
                "numbered_lists": r"^\s*\d+\.\s+",
                "indentation": r"^\s{2,}",
                "blank_lines": r"^\s*$",
            },
            "required_patterns": set(),
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return FormattingRule(
        name=name,
        description=description,
        config=config,
    )

# Export public classes and functions
__all__ = [
    "LengthRule",
    "LengthConfig",
    "LengthValidator",
    "DefaultLengthValidator",
    "ParagraphRule",
    "ParagraphConfig",
    "ParagraphValidator",
    "DefaultParagraphValidator",
    "StyleRule",
    "StyleConfig",
    "StyleValidator",
    "DefaultStyleValidator",
    "FormattingRule",
    "FormattingConfig",
    "FormattingValidator",
    "DefaultFormattingValidator",
    "create_length_rule",
    "create_paragraph_rule",
    "create_style_rule",
    "create_formatting_rule",
]
