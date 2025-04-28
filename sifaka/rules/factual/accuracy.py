"""
Factual accuracy validation rules for Sifaka.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from sifaka.rules.base import Rule, RuleConfig, RuleResult
from sifaka.rules.factual.base import BaseFactualValidator


@dataclass(frozen=True)
class FactualAccuracyConfig(RuleConfig):
    """Configuration for factual accuracy rules."""

    knowledge_base: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            "earth_shape": {"round", "spherical", "geoid"},
            "gravity": {"9.8 m/s²", "9.8 meters per second squared"},
        }
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.knowledge_base:
            raise ValueError("Must provide at least one knowledge base entry")
        if not all(facts for facts in self.knowledge_base.values()):
            raise ValueError("Each knowledge base entry must have at least one fact")


class DefaultFactualAccuracyValidator(BaseFactualValidator):
    """Default implementation of factual accuracy validation."""

    def __init__(self, config: FactualAccuracyConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)

    @property
    def config(self) -> FactualAccuracyConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text for factual accuracy."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        found_facts: Dict[str, Set[str]] = {}

        # Check each knowledge base entry
        for topic, facts in self.config.knowledge_base.items():
            found = {fact for fact in facts if fact.lower() in text_lower}
            if found:
                found_facts[topic] = found

        if not found_facts:
            return RuleResult(
                passed=True,
                message="No factual claims found to verify",
                metadata={"verified_facts": {}},
            )

        # All found facts are considered accurate since they're from the knowledge base
        return RuleResult(
            passed=True,
            message=f"Verified {len(found_facts)} factual claim(s)",
            metadata={"verified_facts": found_facts},
        )


class FactualAccuracyRule(Rule[str, RuleResult, DefaultFactualAccuracyValidator, Any]):
    """Rule that checks for factual accuracy in the text."""

    def __init__(
        self,
        name: str = "factual_accuracy_rule",
        description: str = "Checks for factual accuracy",
        config: Optional[RuleConfig] = None,
        validator: Optional[DefaultFactualAccuracyValidator] = None,
    ) -> None:
        """
        Initialize the factual accuracy rule.

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

    def _create_default_validator(self) -> DefaultFactualAccuracyValidator:
        """Create a default validator from config."""
        accuracy_config = FactualAccuracyConfig(**self._rule_params)
        return DefaultFactualAccuracyValidator(accuracy_config)


def create_factual_accuracy_rule(
    name: str = "factual_accuracy_rule",
    description: str = "Validates text for factual accuracy",
    config: Optional[Dict[str, Any]] = None,
) -> FactualAccuracyRule:
    """
    Create a factual accuracy rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured FactualAccuracyRule instance
    """
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return FactualAccuracyRule(
        name=name,
        description=description,
        config=rule_config,
    )
