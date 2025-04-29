"""
Factual consistency validation rules for Sifaka.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.factual.base import BaseFactualValidator


@dataclass(frozen=True)
class FactualConsistencyConfig(RuleConfig):
    """Configuration for factual consistency rules."""

    contradiction_indicators: List[str] = field(
        default_factory=lambda: [
            "but",
            "however",
            "although",
            "nevertheless",
            "on the other hand",
            "in contrast",
            "despite",
            "yet",
            "while",
            "whereas",
        ]
    )
    confidence_threshold: float = 0.7
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if not self.contradiction_indicators:
            raise ValueError("Must provide at least one contradiction indicator")


class DefaultFactualConsistencyValidator(BaseFactualValidator):
    """Default implementation of factual consistency validation."""

    def __init__(self, config: FactualConsistencyConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)

    @property
    def config(self) -> FactualConsistencyConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for factual consistency."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Check for contradictions using indicators
        found_contradictions = []
        for indicator in self.config.contradiction_indicators:
            if indicator.lower() in text.lower():
                found_contradictions.append(indicator)

        # Calculate confidence score based on contradictions
        confidence_score = 1.0 - (
            len(found_contradictions) / len(self.config.contradiction_indicators)
        )
        meets_threshold = confidence_score >= self.config.confidence_threshold

        if not meets_threshold:
            return RuleResult(
                passed=False,
                message=f"Found potential contradictions: {', '.join(found_contradictions)}",
                metadata={
                    "found_contradictions": found_contradictions,
                    "confidence_score": confidence_score,
                    "threshold": self.config.confidence_threshold,
                },
            )

        return RuleResult(
            passed=True,
            message="No contradictions found",
            metadata={
                "found_contradictions": [],
                "confidence_score": confidence_score,
                "threshold": self.config.confidence_threshold,
            },
        )


class FactualConsistencyRule(Rule[str, RuleResult, DefaultFactualConsistencyValidator, Any]):
    """Rule that checks for factual consistency within the text."""

    def __init__(
        self,
        name: str = "factual_consistency_rule",
        description: str = "Checks for factual consistency",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultFactualConsistencyValidator] = None,
    ) -> None:
        """
        Initialize the factual consistency rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Rule configuration
            validator: Optional custom validator implementation
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

    def _create_default_validator(self) -> DefaultFactualConsistencyValidator:
        """Create a default validator from config."""
        factual_config = FactualConsistencyConfig(**self._rule_params)
        return DefaultFactualConsistencyValidator(factual_config)


def create_factual_consistency_rule(
    name: str = "factual_consistency_rule",
    description: str = "Validates text for factual consistency",
    config: Optional[Dict[str, Any]] = None,
) -> FactualConsistencyRule:
    """
    Create a factual consistency rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured FactualConsistencyRule instance
    """
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return FactualConsistencyRule(
        name=name,
        description=description,
        config=rule_config,
    )
