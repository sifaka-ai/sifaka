"""
Reflector module for Sifaka.

This module provides rules for text reflection and transformation,
including text mirroring, symmetry validation, and pattern matching.
"""

from typing import Dict, List, Optional, Protocol, runtime_checkable, Final, Any
from dataclasses import dataclass, field
from sifaka.rules.base import Rule, RuleResult


@dataclass(frozen=True)
class ReflectionConfig:
    """Configuration for text reflection."""

    mirror_mode: str = "horizontal"  # horizontal, vertical, or both
    preserve_whitespace: bool = True
    preserve_case: bool = True
    ignore_punctuation: bool = False
    symmetry_threshold: float = 1.0  # 1.0 means perfect symmetry
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        valid_modes = {"horizontal", "vertical", "both"}
        if self.mirror_mode not in valid_modes:
            raise ValueError(f"Mirror mode must be one of {valid_modes}")
        if not 0.0 <= self.symmetry_threshold <= 1.0:
            raise ValueError("Symmetry threshold must be between 0.0 and 1.0")
        if self.cache_size < 0:
            raise ValueError("Cache size must be non-negative")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.cost < 0:
            raise ValueError("Cost must be non-negative")


@dataclass(frozen=True)
class PatternConfig:
    """Configuration for pattern matching and transformation."""

    pattern_type: str = "repeat"  # repeat, alternate, or custom
    pattern_length: int = 2
    custom_pattern: Optional[str] = None
    case_sensitive: bool = True
    allow_overlap: bool = False
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        valid_types = {"repeat", "alternate", "custom"}
        if self.pattern_type not in valid_types:
            raise ValueError(f"Pattern type must be one of {valid_types}")
        if self.pattern_length < 1:
            raise ValueError("Pattern length must be positive")
        if self.pattern_type == "custom" and not self.custom_pattern:
            raise ValueError("Custom pattern must be provided for custom pattern type")
        if self.cache_size < 0:
            raise ValueError("Cache size must be non-negative")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.cost < 0:
            raise ValueError("Cost must be non-negative")


@runtime_checkable
class ReflectionValidator(Protocol):
    """Protocol for text reflection validation."""

    @property
    def config(self) -> ReflectionConfig:
        """Get validator configuration."""
        ...

    def validate(self, text: str) -> RuleResult:
        """Validate reflected text."""
        ...


@runtime_checkable
class PatternValidator(Protocol):
    """Protocol for pattern validation."""

    @property
    def config(self) -> PatternConfig:
        """Get validator configuration."""
        ...

    def validate(self, text: str) -> RuleResult:
        """Validate pattern in text."""
        ...


class DefaultReflectionValidator:
    """Default implementation of text reflection validation."""

    def __init__(self, config: ReflectionConfig):
        self._config = config

    @property
    def config(self) -> ReflectionConfig:
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text reflection."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Process text according to configuration
        processed_text = text
        if not self.config.preserve_case:
            processed_text = processed_text.lower()
        if self.config.ignore_punctuation:
            processed_text = "".join(c for c in processed_text if c.isalnum() or c.isspace())

        lines = processed_text.split("\n")
        symmetry_score = 0.0
        reflected_text = ""

        if self.config.mirror_mode in {"horizontal", "both"}:
            # Check horizontal symmetry
            for line in lines:
                if not self.config.preserve_whitespace:
                    line = line.strip()
                reversed_line = line[::-1]
                similarity = self._calculate_similarity(line, reversed_line)
                symmetry_score += similarity
            symmetry_score /= len(lines) if lines else 1

        if self.config.mirror_mode in {"vertical", "both"}:
            # Check vertical symmetry
            reversed_lines = lines[::-1]
            vertical_score = (
                sum(self._calculate_similarity(l1, l2) for l1, l2 in zip(lines, reversed_lines))
                / len(lines)
                if lines
                else 0
            )

            if self.config.mirror_mode == "both":
                symmetry_score = (symmetry_score + vertical_score) / 2
            else:
                symmetry_score = vertical_score

        is_symmetric = symmetry_score >= self.config.symmetry_threshold

        return RuleResult(
            passed=is_symmetric,
            message=(
                "Text meets symmetry requirements"
                if is_symmetric
                else "Text does not meet symmetry requirements"
            ),
            metadata={
                "symmetry_score": symmetry_score,
                "mirror_mode": self.config.mirror_mode,
                "original_text": text,
                "processed_text": processed_text,
            },
        )

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two strings."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        matches = sum(1 for a, b in zip(text1, text2) if a == b)
        return matches / max(len(text1), len(text2))


class DefaultPatternValidator:
    """Default implementation of pattern validation."""

    def __init__(self, config: PatternConfig):
        self._config = config

    @property
    def config(self) -> PatternConfig:
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate pattern in text."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        processed_text = text if self.config.case_sensitive else text.lower()
        pattern_found = False
        matches = []

        if self.config.pattern_type == "repeat":
            # Check for repeating patterns
            for i in range(len(processed_text) - self.config.pattern_length + 1):
                if not self.config.allow_overlap and i < len(matches):
                    continue
                pattern = processed_text[i : i + self.config.pattern_length]
                if processed_text.count(pattern) > 1:
                    pattern_found = True
                    matches.append(pattern)

        elif self.config.pattern_type == "alternate":
            # Check for alternating patterns
            for i in range(len(processed_text) - self.config.pattern_length * 2 + 1):
                if not self.config.allow_overlap and i < len(matches):
                    continue
                pattern1 = processed_text[i : i + self.config.pattern_length]
                pattern2 = processed_text[
                    i + self.config.pattern_length : i + self.config.pattern_length * 2
                ]
                if pattern1 != pattern2 and processed_text.count(pattern1 + pattern2) > 1:
                    pattern_found = True
                    matches.append(pattern1 + pattern2)

        else:  # custom pattern
            if self.config.custom_pattern:
                pattern = self.config.custom_pattern
                if not self.config.case_sensitive:
                    pattern = pattern.lower()
                pattern_found = pattern in processed_text
                matches = [pattern] if pattern_found else []

        return RuleResult(
            passed=pattern_found,
            message=(
                "Pattern requirements met" if pattern_found else "Pattern requirements not met"
            ),
            metadata={
                "pattern_type": self.config.pattern_type,
                "matches_found": matches,
                "match_count": len(matches),
                "original_text": text,
                "processed_text": processed_text,
            },
        )


class ReflectionRule(Rule):
    """Rule for validating text reflection."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: ReflectionValidator,
    ) -> None:
        super().__init__(name=name, description=description)
        self._validator = validator

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output reflection."""
        return self._validator.validate(output)


class PatternRule(Rule):
    """Rule for validating text patterns."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: PatternValidator,
    ) -> None:
        super().__init__(name=name, description=description)
        self._validator = validator

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output pattern."""
        return self._validator.validate(output)


# Default configurations for common use cases
DEFAULT_REFLECTION_CONFIGS: Final[Dict[str, Dict[str, Any]]] = {
    "palindrome": {
        "mirror_mode": "horizontal",
        "preserve_whitespace": False,
        "preserve_case": False,
        "ignore_punctuation": True,
        "symmetry_threshold": 1.0,
    },
    "visual_mirror": {
        "mirror_mode": "both",
        "preserve_whitespace": True,
        "preserve_case": True,
        "ignore_punctuation": False,
        "symmetry_threshold": 0.8,
    },
    "partial_symmetry": {
        "mirror_mode": "horizontal",
        "preserve_whitespace": True,
        "preserve_case": True,
        "ignore_punctuation": False,
        "symmetry_threshold": 0.6,
    },
}

# Default pattern configurations
DEFAULT_PATTERN_CONFIGS: Final[Dict[str, Dict[str, Any]]] = {
    "word_repeat": {
        "pattern_type": "repeat",
        "pattern_length": 4,
        "case_sensitive": False,
        "allow_overlap": False,
    },
    "character_alternate": {
        "pattern_type": "alternate",
        "pattern_length": 1,
        "case_sensitive": True,
        "allow_overlap": True,
    },
    "custom_sequence": {
        "pattern_type": "custom",
        "custom_pattern": "abcabc",
        "case_sensitive": False,
        "allow_overlap": False,
    },
}


def create_reflection_rule(
    name: str,
    description: str,
    mirror_mode: str = "horizontal",
    preserve_whitespace: bool = True,
    preserve_case: bool = True,
    ignore_punctuation: bool = False,
    symmetry_threshold: float = 1.0,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
) -> ReflectionRule:
    """Create a reflection rule with default configuration."""
    config = ReflectionConfig(
        mirror_mode=mirror_mode,
        preserve_whitespace=preserve_whitespace,
        preserve_case=preserve_case,
        ignore_punctuation=ignore_punctuation,
        symmetry_threshold=symmetry_threshold,
        cache_size=cache_size,
        priority=priority,
        cost=cost,
    )
    validator = DefaultReflectionValidator(config)
    return ReflectionRule(name=name, description=description, validator=validator)


def create_pattern_rule(
    name: str,
    description: str,
    pattern_type: str = "repeat",
    pattern_length: int = 2,
    custom_pattern: Optional[str] = None,
    case_sensitive: bool = True,
    allow_overlap: bool = False,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
) -> PatternRule:
    """Create a pattern rule with default configuration."""
    config = PatternConfig(
        pattern_type=pattern_type,
        pattern_length=pattern_length,
        custom_pattern=custom_pattern,
        case_sensitive=case_sensitive,
        allow_overlap=allow_overlap,
        cache_size=cache_size,
        priority=priority,
        cost=cost,
    )
    validator = DefaultPatternValidator(config)
    return PatternRule(name=name, description=description, validator=validator)


class Reflector:
    """Main class for text reflection and pattern matching functionality."""

    def __init__(
        self,
        reflection_config: Optional[ReflectionConfig] = None,
        pattern_config: Optional[PatternConfig] = None,
    ) -> None:
        """Initialize the reflector with optional configurations."""
        self._reflection_config = reflection_config or ReflectionConfig()
        self._pattern_config = pattern_config or PatternConfig()

        # Create default validators
        self._reflection_validator = DefaultReflectionValidator(self._reflection_config)
        self._pattern_validator = DefaultPatternValidator(self._pattern_config)

        # Create rules
        self._reflection_rule = ReflectionRule(
            name="Text Reflection",
            description="Validates text reflection and symmetry",
            validator=self._reflection_validator,
        )
        self._pattern_rule = PatternRule(
            name="Pattern Matching",
            description="Validates text patterns and transformations",
            validator=self._pattern_validator,
        )

    @property
    def reflection_config(self) -> ReflectionConfig:
        """Get the reflection configuration."""
        return self._reflection_config

    @property
    def pattern_config(self) -> PatternConfig:
        """Get the pattern configuration."""
        return self._pattern_config

    def validate_reflection(self, text: str) -> RuleResult:
        """Validate text reflection and symmetry."""
        return self._reflection_rule.validate(text)

    def validate_pattern(self, text: str) -> RuleResult:
        """Validate text patterns."""
        return self._pattern_rule.validate(text)

    def validate(self, text: str) -> Dict[str, RuleResult]:
        """Validate both reflection and patterns."""
        return {
            "reflection": self.validate_reflection(text),
            "pattern": self.validate_pattern(text),
        }


# Export public classes and functions
__all__ = [
    "Reflector",
    "ReflectionConfig",
    "PatternConfig",
    "ReflectionValidator",
    "PatternValidator",
    "ReflectionRule",
    "PatternRule",
    "create_reflection_rule",
    "create_pattern_rule",
]
