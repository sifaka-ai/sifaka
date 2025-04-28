"""
Repetition pattern validation configuration.

This module provides configuration classes for customizing repetition pattern validation,
including pattern types, lengths, and matching options.
"""

from dataclasses import dataclass
from typing import Optional

from sifaka.rules.base import RuleConfig


@dataclass(frozen=True)
class RepetitionConfig(RuleConfig):
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
        super().__post_init__()
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
