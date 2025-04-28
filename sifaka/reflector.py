"""
DEPRECATED: This module has been replaced by pattern_rules.py

The functionality previously provided by the Reflector class has been moved to
specialized pattern detection rules in sifaka.rules.pattern_rules.

Instead of using this module, use the new pattern rules:

```python
from sifaka.rules.pattern_rules import SymmetryRule, RepetitionRule

# For symmetry detection (previously horizontal/vertical reflection)
symmetry_rule = SymmetryRule(
    name="symmetry_check",
    config=SymmetryConfig(
        mirror_mode="both",
        symmetry_threshold=0.8
    )
)

# For pattern detection (previously pattern matching)
repetition_rule = RepetitionRule(
    name="repetition_check",
    config=RepetitionConfig(
        pattern_type="repeat",
        pattern_length=3
    )
)
```

This module will be removed in a future version.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Union

from sifaka.rules.base import RuleConfig
from sifaka.rules.pattern_rules import (
    RepetitionConfig,
    RepetitionRule,
    RuleResult,
    SymmetryConfig,
    SymmetryRule,
)


@dataclass(frozen=True)
class PatternConfig(RuleConfig):
    """Configuration for pattern detection."""

    pattern_type: Literal["repeat", "alternate", "custom"] = "repeat"
    pattern_length: int = 2
    custom_pattern: Optional[str] = None
    case_sensitive: bool = True
    allow_overlap: bool = False
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if self.pattern_type not in ["repeat", "alternate", "custom"]:
            raise ValueError(
                f"Pattern type must be one of: repeat, alternate, custom, got {self.pattern_type}"
            )
        if self.pattern_length < 1:
            raise ValueError("Pattern length must be positive")
        if self.pattern_type == "custom" and not self.custom_pattern:
            raise ValueError("Custom pattern must be provided when pattern_type is 'custom'")


@dataclass(frozen=True)
class ReflectionConfig(RuleConfig):
    """Configuration for reflection detection."""

    mirror_mode: Literal["horizontal", "vertical", "both"] = "horizontal"
    preserve_whitespace: bool = True
    preserve_case: bool = True
    ignore_punctuation: bool = False
    symmetry_threshold: float = 1.0
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if self.mirror_mode not in ["horizontal", "vertical", "both"]:
            raise ValueError(
                f"Mirror mode must be one of: horizontal, vertical, both, got {self.mirror_mode}"
            )
        if not 0.0 <= self.symmetry_threshold <= 1.0:
            raise ValueError("Symmetry threshold must be between 0.0 and 1.0")


class PatternRule(RepetitionRule):
    """Legacy pattern rule for backward compatibility."""

    def __init__(
        self,
        name: str = "pattern_rule",
        description: str = "Detects patterns in text",
        config: Optional[PatternConfig] = None,
    ) -> None:
        """Initialize with warning about deprecation."""
        warnings.warn(
            "PatternRule is deprecated and will be removed in a future version. "
            "Use RepetitionRule from sifaka.rules.pattern_rules instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Convert PatternConfig to RepetitionConfig
        if config is None:
            config = PatternConfig()

        repetition_config = RepetitionConfig(
            pattern_type=config.pattern_type,
            pattern_length=config.pattern_length,
            custom_pattern=config.custom_pattern,
            case_sensitive=config.case_sensitive,
            allow_overlap=config.allow_overlap,
            cache_size=config.cache_size,
            priority=config.priority,
            cost=config.cost,
        )

        super().__init__(name=name, description=description, config=repetition_config)


class ReflectionRule(SymmetryRule):
    """Legacy reflection rule for backward compatibility."""

    def __init__(
        self,
        name: str = "reflection_rule",
        description: str = "Detects symmetry in text",
        config: Optional[ReflectionConfig] = None,
    ) -> None:
        """Initialize with warning about deprecation."""
        warnings.warn(
            "ReflectionRule is deprecated and will be removed in a future version. "
            "Use SymmetryRule from sifaka.rules.pattern_rules instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Convert ReflectionConfig to SymmetryConfig
        if config is None:
            config = ReflectionConfig()

        symmetry_config = SymmetryConfig(
            mirror_mode=config.mirror_mode,
            preserve_whitespace=config.preserve_whitespace,
            preserve_case=config.preserve_case,
            ignore_punctuation=config.ignore_punctuation,
            symmetry_threshold=config.symmetry_threshold,
            cache_size=config.cache_size,
            priority=config.priority,
            cost=config.cost,
        )

        super().__init__(name=name, description=description, config=symmetry_config)


# Default reflection configurations
DEFAULT_REFLECTION_CONFIGS = {
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
    "partial_mirror": {
        "mirror_mode": "horizontal",
        "preserve_whitespace": True,
        "preserve_case": True,
        "ignore_punctuation": False,
        "symmetry_threshold": 0.6,
    },
}

# Default pattern configurations
DEFAULT_PATTERN_CONFIGS = {
    "word_repeat": {
        "pattern_type": "repeat",
        "pattern_length": 4,
        "case_sensitive": False,
        "allow_overlap": False,
    },
    "character_repeat": {
        "pattern_type": "repeat",
        "pattern_length": 1,
        "case_sensitive": True,
        "allow_overlap": True,
    },
    "character_alternate": {
        "pattern_type": "alternate",
        "pattern_length": 1,
        "case_sensitive": False,
        "allow_overlap": False,
    },
    "phrase_repeat": {
        "pattern_type": "custom",
        "custom_pattern": "the",
        "case_sensitive": False,
        "allow_overlap": False,
    },
}


def create_pattern_rule(
    name: str = "pattern_rule",
    description: str = "Detects patterns in text",
    pattern_type: str = "repeat",
    pattern_length: int = 2,
    custom_pattern: Optional[str] = None,
    case_sensitive: bool = True,
    allow_overlap: bool = False,
) -> PatternRule:
    """
    Create a pattern rule with the given configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        pattern_type: Type of pattern to detect (repeat, alternate, custom)
        pattern_length: Length of pattern to detect
        custom_pattern: Custom pattern to detect (required if pattern_type is 'custom')
        case_sensitive: Whether to be case sensitive
        allow_overlap: Whether to allow overlapping patterns

    Returns:
        PatternRule: Configured pattern rule
    """
    config = PatternConfig(
        pattern_type=pattern_type,
        pattern_length=pattern_length,
        custom_pattern=custom_pattern,
        case_sensitive=case_sensitive,
        allow_overlap=allow_overlap,
    )
    return PatternRule(name=name, description=description, config=config)


def create_reflection_rule(
    name: str = "reflection_rule",
    description: str = "Detects symmetry in text",
    mirror_mode: str = "horizontal",
    preserve_whitespace: bool = True,
    preserve_case: bool = True,
    ignore_punctuation: bool = False,
    symmetry_threshold: float = 1.0,
) -> ReflectionRule:
    """
    Create a reflection rule with the given configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        mirror_mode: Type of symmetry to detect (horizontal, vertical, both)
        preserve_whitespace: Whether to preserve whitespace
        preserve_case: Whether to preserve case
        ignore_punctuation: Whether to ignore punctuation
        symmetry_threshold: Threshold for symmetry detection

    Returns:
        ReflectionRule: Configured reflection rule
    """
    config = ReflectionConfig(
        mirror_mode=mirror_mode,
        preserve_whitespace=preserve_whitespace,
        preserve_case=preserve_case,
        ignore_punctuation=ignore_punctuation,
        symmetry_threshold=symmetry_threshold,
    )
    return ReflectionRule(name=name, description=description, config=config)


class Reflector:
    """
    DEPRECATED: Use SymmetryRule and RepetitionRule instead.

    This class has been replaced by specialized pattern rules in
    sifaka.rules.pattern_rules.
    """

    def __init__(
        self,
        reflection_config: Optional[SymmetryConfig] = None,
        pattern_config: Optional[RepetitionConfig] = None,
    ) -> None:
        """Initialize with warning about deprecation."""
        warnings.warn(
            "The Reflector class is deprecated and will be removed in a future version. "
            "Use SymmetryRule and RepetitionRule from sifaka.rules.pattern_rules instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self._symmetry_rule = SymmetryRule(
            name="symmetry_validator",
            config=reflection_config or SymmetryConfig(),
        )

        self._pattern_rule = RepetitionRule(
            name="pattern_detector",
            config=pattern_config or RepetitionConfig(),
        )

    @property
    def reflection_config(self) -> SymmetryConfig:
        """Get symmetry configuration."""
        return self._symmetry_rule.config

    @property
    def pattern_config(self) -> RepetitionConfig:
        """Get pattern configuration."""
        return self._pattern_rule.config

    def validate_reflection(self, text: str) -> RuleResult:
        """Validate text symmetry (deprecated)."""
        return self._symmetry_rule.validate(text)

    def validate_pattern(self, text: str) -> RuleResult:
        """Validate text patterns (deprecated)."""
        return self._pattern_rule.validate(text)

    def validate(self, text: str) -> Dict[str, RuleResult]:
        """Run all validations (deprecated)."""
        return {
            "reflection": self.validate_reflection(text),
            "pattern": self.validate_pattern(text),
        }
