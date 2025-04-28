"""
Repetition pattern validation rules.

This module provides rule classes for validating repetition patterns in text,
integrating configuration and validation logic.
"""

from typing import Any, Optional

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from .config import RepetitionConfig
from .validator import RepetitionValidator


class RepetitionRule(Rule[str, RuleResult, RepetitionValidator, Any]):
    """Rule that validates text repetition patterns."""

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
