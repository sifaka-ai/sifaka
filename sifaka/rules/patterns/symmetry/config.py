"""
Symmetry pattern validation configuration.

This module provides configuration classes for customizing symmetry pattern validation,
including mirror modes, whitespace handling, and similarity thresholds.
"""

from dataclasses import dataclass

from sifaka.rules.base import RuleConfig


@dataclass(frozen=True)
class SymmetryConfig(RuleConfig):
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
        super().__post_init__()
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
