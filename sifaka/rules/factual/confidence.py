"""
Confidence validation rules for Sifaka.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sifaka.rules.base import Rule, RuleConfig, RuleResult, RuleValidator
from sifaka.rules.factual.base import BaseFactualValidator


@dataclass(frozen=True)
class ConfidenceConfig(RuleConfig):
    """Configuration for confidence rules."""

    confidence_indicators: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "high": ["definitely", "certainly", "always", "never", "must", "will"],
            "medium": ["likely", "probably", "usually", "often", "generally"],
            "low": ["maybe", "possibly", "sometimes", "occasionally", "might"],
            "uncertain": ["perhaps", "could", "may", "seems", "appears"],
        }
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.confidence_indicators:
            raise ValueError("Must provide at least one confidence level")
        if not all(indicators for indicators in self.confidence_indicators.values()):
            raise ValueError("Each confidence level must have at least one indicator")


class DefaultConfidenceValidator(BaseFactualValidator):
    """Default implementation of confidence validation."""

    def __init__(self, config: ConfidenceConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)

    @property
    def config(self) -> ConfidenceConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for confidence indicators."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        confidence_scores: Dict[str, List[str]] = {}

        # Find indicators for each confidence level
        for level, indicators in self.config.confidence_indicators.items():
            found = [ind for ind in indicators if ind.lower() in text_lower]
            if found:
                confidence_scores[level] = found

        if not confidence_scores:
            return RuleResult(
                passed=False,
                message="No confidence indicators found",
                metadata={"confidence_levels": {}},
            )

        # Determine dominant confidence level
        dominant_level = max(confidence_scores.items(), key=lambda x: len(x[1]))[0]

        return RuleResult(
            passed=True,
            message=f"Found {dominant_level} confidence level",
            metadata={
                "confidence_levels": confidence_scores,
                "dominant_level": dominant_level,
            },
        )


class ConfidenceRule(Rule[str, RuleResult, DefaultConfidenceValidator, Any]):
    """Rule that checks for confidence indicators in the text."""

    def __init__(
        self,
        name: str = "confidence_rule",
        description: str = "Checks for confidence indicators",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultConfidenceValidator] = None,
    ) -> None:
        """
        Initialize the confidence rule.

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

    def _create_default_validator(self) -> DefaultConfidenceValidator:
        """Create a default validator from config."""
        confidence_config = ConfidenceConfig(**self._rule_params)
        return DefaultConfidenceValidator(confidence_config)


def create_confidence_rule(
    name: str = "confidence_rule",
    description: str = "Validates text for confidence indicators",
    config: Optional[Dict[str, Any]] = None,
) -> ConfidenceRule:
    """
    Create a confidence rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured ConfidenceRule instance
    """
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return ConfidenceRule(
        name=name,
        description=description,
        config=rule_config,
    )
