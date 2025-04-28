"""
Symmetry pattern validation rules.

This module provides rule classes for validating symmetry patterns in text,
integrating configuration and validation logic.
"""

from typing import Any, Optional

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from .config import SymmetryConfig
from .validator import SymmetryValidator


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
