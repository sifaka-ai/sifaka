"""
Pattern-based validation rules for Sifaka.

This module provides specialized rules for detecting and validating patterns in text,
including symmetry detection and repetition analysis.
"""

from dataclasses import dataclass
from typing import Any, Optional

from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult


@dataclass(frozen=True)
class SymmetryConfig:
    """Configuration for text symmetry validation."""

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
class RepetitionConfig:
    """Configuration for pattern repetition detection."""

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


class SymmetryValidator(BaseValidator[str]):
    """Validator for text symmetry patterns."""

    def __init__(self, config: SymmetryConfig) -> None:
        """Initialize with symmetry configuration."""
        self._config = config

    @property
    def config(self) -> SymmetryConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text symmetry."""
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


class SymmetryRule(Rule[str, RuleResult, SymmetryValidator, Any]):
    """Rule that validates text symmetry patterns."""

    def __init__(
        self,
        name: str = "symmetry_validator",
        description: str = "Validates text symmetry patterns",
        config: Optional[RuleConfig] = None,
        validator: Optional[SymmetryValidator] = None,
    ) -> None:
        """Initialize the symmetry rule."""
        # Store symmetry parameters for creating the default validator
        self._symmetry_params = {}
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._symmetry_params = {
                "mirror_mode": params_source.get("mirror_mode", "horizontal"),
                "preserve_whitespace": params_source.get("preserve_whitespace", True),
                "preserve_case": params_source.get("preserve_case", True),
                "ignore_punctuation": params_source.get("ignore_punctuation", False),
                "symmetry_threshold": params_source.get("symmetry_threshold", 1.0),
                "cache_size": config.cache_size,
                "priority": config.priority.value,
                "cost": config.cost,
            }

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

    def _create_default_validator(self) -> SymmetryValidator:
        """Create a default validator from config."""
        symmetry_config = SymmetryConfig(**self._symmetry_params)
        return SymmetryValidator(symmetry_config)


class RepetitionValidator(BaseValidator[str]):
    """Validator for repetitive patterns in text."""

    def __init__(self, config: RepetitionConfig) -> None:
        """Initialize with repetition configuration."""
        self._config = config

    @property
    def config(self) -> RepetitionConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for repetitive patterns."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Process text according to configuration
        if not self.config.case_sensitive:
            text = text.lower()

        matches = []
        match_count = 0

        if self.config.pattern_type == "custom" and self.config.custom_pattern:
            # Use custom pattern
            pattern = self.config.custom_pattern
            matches = self._find_pattern_matches(text, pattern)
            match_count = len(matches)
        else:
            # Find repeating or alternating patterns
            for i in range(len(text) - self.config.pattern_length + 1):
                if not self.config.allow_overlap and any(i < end for _, end in matches):
                    continue

                pattern = text[i : i + self.config.pattern_length]
                if self._is_valid_pattern(text, pattern, i):
                    matches.append((i, i + len(pattern)))
                    match_count += 1

        return RuleResult(
            passed=True,  # Pattern detection always passes, just reports findings
            message=f"Found {match_count} pattern matches",
            metadata={
                "match_count": match_count,
                "matches_found": [
                    text[start:end] for start, end in matches[:10]
                ],  # Limit to first 10
                "pattern_type": self.config.pattern_type,
                "pattern_length": self.config.pattern_length,
            },
        )

    def _is_valid_pattern(self, text: str, pattern: str, start_pos: int) -> bool:
        """Check if a pattern is valid according to configuration."""
        if self.config.pattern_type == "repeat":
            # Look for exact repetitions
            return text.count(pattern) > 1
        elif self.config.pattern_type == "alternate":
            # Look for alternating patterns
            if len(text) < start_pos + len(pattern) * 2:
                return False
            next_chunk = text[start_pos + len(pattern) : start_pos + len(pattern) * 2]
            return pattern != next_chunk and text.count(pattern + next_chunk) > 1
        return False

    def _find_pattern_matches(self, text: str, pattern: str) -> list[tuple[int, int]]:
        """Find all occurrences of a pattern in text."""
        matches = []
        start = 0
        while True:
            pos = text.find(pattern, start)
            if pos == -1:
                break
            matches.append((pos, pos + len(pattern)))
            start = pos + (1 if self.config.allow_overlap else len(pattern))
        return matches


class RepetitionRule(Rule[str, RuleResult, RepetitionValidator, Any]):
    """Rule that detects repetitive patterns in text."""

    def __init__(
        self,
        name: str = "repetition_detector",
        description: str = "Detects repetitive patterns in text",
        config: Optional[RuleConfig] = None,
        validator: Optional[RepetitionValidator] = None,
    ) -> None:
        """Initialize the repetition rule."""
        # Store repetition parameters for creating the default validator
        self._repetition_params = {}
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._repetition_params = {
                "pattern_type": params_source.get("pattern_type", "repeat"),
                "pattern_length": params_source.get("pattern_length", 2),
                "custom_pattern": params_source.get("custom_pattern"),
                "case_sensitive": params_source.get("case_sensitive", True),
                "allow_overlap": params_source.get("allow_overlap", False),
                "cache_size": config.cache_size,
                "priority": config.priority.value,
                "cost": config.cost,
            }

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

    def _create_default_validator(self) -> RepetitionValidator:
        """Create a default validator from config."""
        repetition_config = RepetitionConfig(**self._repetition_params)
        return RepetitionValidator(repetition_config)


# Export public classes and functions
__all__ = [
    "SymmetryRule",
    "RepetitionRule",
    "SymmetryConfig",
    "RepetitionConfig",
    "SymmetryValidator",
    "RepetitionValidator",
]
