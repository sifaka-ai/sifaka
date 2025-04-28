"""
Fact-checking rules for Sifaka.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from sifaka.rules.base import BaseValidator, Rule, RuleConfig, RuleResult


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


@dataclass(frozen=True)
class CitationConfig(RuleConfig):
    """Configuration for citation rules."""

    citation_patterns: List[str] = field(
        default_factory=lambda: [
            r"\[[\d]+\]",  # [1], [2], etc.
            r"\([A-Za-z]+ et al., \d{4}\)",  # (Smith et al., 2020)
            r"\([A-Za-z]+, \d{4}\)",  # (Smith, 2020)
            r"https?://[^\s]+",  # URLs
        ]
    )
    required_citations: bool = True
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.citation_patterns:
            raise ValueError("Must provide at least one citation pattern")


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


class DefaultFactualConsistencyValidator(BaseValidator[str]):
    """Default implementation of factual consistency validation."""

    def __init__(self, config: FactualConsistencyConfig) -> None:
        """Initialize with configuration."""
        self._config = config

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


class DefaultConfidenceValidator(RuleValidator[str]):
    """Default implementation of confidence validation."""

    def __init__(self, config: ConfidenceConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> ConfidenceConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
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

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str


class DefaultCitationValidator(RuleValidator[str]):
    """Default implementation of citation validation."""

    def __init__(self, config: CitationConfig) -> None:
        """Initialize with configuration."""
        self._config = config
        self._patterns = [re.compile(pattern) for pattern in config.citation_patterns]

    @property
    def config(self) -> CitationConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text for citations."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        found_citations = []
        for pattern in self._patterns:
            found_citations.extend(pattern.findall(text))

        if not found_citations and self.config.required_citations:
            return RuleResult(
                passed=False,
                message="No citations found",
                metadata={
                    "found_citations": [],
                    "required": self.config.required_citations,
                },
            )

        return RuleResult(
            passed=True,
            message=f"Found {len(found_citations)} citation(s)",
            metadata={
                "found_citations": found_citations,
                "required": self.config.required_citations,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str


class DefaultFactualAccuracyValidator(RuleValidator[str]):
    """Default implementation of factual accuracy validation."""

    def __init__(self, config: FactualAccuracyConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> FactualAccuracyConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
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

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str


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


class ConfidenceRule(Rule):
    """Rule that checks for confidence indicators in the text."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the confidence rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        confidence_config = ConfidenceConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultConfidenceValidator(confidence_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output for confidence indicators."""
        return self._validator.validate(output)


class CitationRule(Rule):
    """Rule that checks for citations in the text."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the citation rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        citation_config = CitationConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultCitationValidator(citation_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output for citations."""
        return self._validator.validate(output)


class FactualAccuracyRule(Rule):
    """Rule that checks for factual accuracy in the text."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the factual accuracy rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        accuracy_config = FactualAccuracyConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultFactualAccuracyValidator(accuracy_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output for factual accuracy."""
        return self._validator.validate(output)


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


def create_citation_rule(
    name: str = "citation_rule",
    description: str = "Validates text for citations",
    config: Optional[Dict[str, Any]] = None,
) -> CitationRule:
    """
    Create a citation rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured CitationRule instance
    """
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return CitationRule(
        name=name,
        description=description,
        config=rule_config,
    )


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


# Export public classes and functions
__all__ = [
    "FactualConsistencyRule",
    "FactualConsistencyConfig",
    "DefaultFactualConsistencyValidator",
    "ConfidenceRule",
    "ConfidenceConfig",
    "DefaultConfidenceValidator",
    "CitationRule",
    "CitationConfig",
    "DefaultCitationValidator",
    "FactualAccuracyRule",
    "FactualAccuracyConfig",
    "DefaultFactualAccuracyValidator",
    "create_factual_consistency_rule",
    "create_confidence_rule",
    "create_citation_rule",
    "create_factual_accuracy_rule",
]
