"""
Safety-related rules for Sifaka.
"""

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable, Final
from dataclasses import dataclass, field
from sifaka.rules.base import Rule, RuleResult, RuleValidator, RuleConfig


@dataclass(frozen=True)
class ToxicityConfig(RuleConfig):
    """Configuration for toxicity validation."""

    threshold: float = 0.5
    indicators: List[str] = field(
        default_factory=lambda: [
            "hate",
            "offensive",
            "vulgar",
            "profanity",
            "obscene",
            "racist",
            "sexist",
            "discriminatory",
            "threatening",
            "harassing",
        ]
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        if not self.indicators:
            raise ValueError("Must provide at least one toxicity indicator")


@dataclass(frozen=True)
class BiasConfig(RuleConfig):
    """Configuration for bias validation."""

    threshold: float = 0.3
    categories: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "gender": [
                "sexist",
                "misogynistic",
                "patriarchal",
                "gender bias",
                "gender stereotype",
            ],
            "racial": [
                "racist",
                "racial bias",
                "ethnic stereotype",
                "racial prejudice",
                "discriminatory",
            ],
            "age": [
                "ageist",
                "age discrimination",
                "age bias",
                "age stereotype",
                "generational bias",
            ],
            "cultural": [
                "cultural bias",
                "xenophobic",
                "ethnocentric",
                "cultural stereotype",
                "cultural prejudice",
            ],
        }
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        if not self.categories:
            raise ValueError("Must provide at least one bias category")
        for category, indicators in self.categories.items():
            if not indicators:
                raise ValueError(f"Category {category} must have at least one indicator")


@dataclass(frozen=True)
class HarmfulContentConfig(RuleConfig):
    """Configuration for harmful content validation."""

    categories: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "violence": [
                "violent",
                "brutal",
                "aggressive",
                "threatening",
                "dangerous",
            ],
            "self_harm": [
                "suicide",
                "self-harm",
                "self-injury",
                "self-destructive",
                "harmful behavior",
            ],
            "exploitation": [
                "exploitative",
                "manipulative",
                "coercive",
                "predatory",
                "abusive",
            ],
            "misinformation": [
                "false",
                "misleading",
                "deceptive",
                "propaganda",
                "disinformation",
            ],
        }
    )
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not self.categories:
            raise ValueError("Must provide at least one harmful content category")
        for category, indicators in self.categories.items():
            if not indicators:
                raise ValueError(f"Category {category} must have at least one indicator")


class DefaultToxicityValidator(RuleValidator[str]):
    """Default implementation of toxicity validation."""

    def __init__(self, config: ToxicityConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> ToxicityConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text for toxicity."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        toxic_indicators = [
            indicator for indicator in self.config.indicators if indicator in text_lower
        ]

        toxicity_score = len(toxic_indicators) / len(self.config.indicators)

        if toxicity_score > self.config.threshold:
            return RuleResult(
                passed=False,
                message=f"Output contains toxic content (score: {toxicity_score:.2f})",
                metadata={"toxicity_score": toxicity_score, "toxic_indicators": toxic_indicators},
            )

        return RuleResult(
            passed=True,
            message="No toxic content detected",
            metadata={"toxicity_score": toxicity_score},
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str


class DefaultBiasValidator(RuleValidator[str]):
    """Default implementation of bias validation."""

    def __init__(self, config: BiasConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> BiasConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text for bias."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        bias_scores: Dict[str, float] = {}
        detected_biases: Dict[str, List[str]] = {}

        for category, indicators in self.config.categories.items():
            found_indicators = [indicator for indicator in indicators if indicator in text_lower]
            if found_indicators:
                bias_scores[category] = len(found_indicators) / len(indicators)
                detected_biases[category] = found_indicators

        if not bias_scores:
            return RuleResult(
                passed=True,
                message="No biased content detected",
                metadata={
                    "bias_scores": {},
                    "detected_biases": {},
                },
            )

        overall_bias_score = sum(bias_scores.values()) / len(self.config.categories)

        if overall_bias_score > self.config.threshold:
            return RuleResult(
                passed=False,
                message=f"Output contains biased content (score: {overall_bias_score:.2f})",
                metadata={
                    "bias_scores": bias_scores,
                    "detected_biases": detected_biases,
                    "overall_score": overall_bias_score,
                },
            )

        return RuleResult(
            passed=True,
            message="No significant bias detected",
            metadata={
                "bias_scores": bias_scores,
                "detected_biases": detected_biases,
                "overall_score": overall_bias_score,
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str


class DefaultHarmfulContentValidator(RuleValidator[str]):
    """Default implementation of harmful content validation."""

    def __init__(self, config: HarmfulContentConfig) -> None:
        """Initialize with configuration."""
        self._config = config

    @property
    def config(self) -> HarmfulContentConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text for harmful content."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        harmful_content: Dict[str, List[str]] = {}

        for category, indicators in self.config.categories.items():
            found_indicators = [indicator for indicator in indicators if indicator in text_lower]
            if found_indicators:
                harmful_content[category] = found_indicators

        if harmful_content:
            return RuleResult(
                passed=False,
                message="Output contains harmful content",
                metadata={
                    "harmful_content": harmful_content,
                    "categories_found": list(harmful_content.keys()),
                },
            )

        return RuleResult(
            passed=True,
            message="No harmful content detected",
            metadata={
                "harmful_content": {},
                "categories_found": [],
            },
        )

    def can_validate(self, output: str) -> bool:
        """Check if this validator can handle the input."""
        return isinstance(output, str)

    @property
    def validation_type(self) -> type[str]:
        """Get the type of input this validator can handle."""
        return str


class ToxicityRule(Rule):
    """Rule for validating text toxicity."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the rule with toxicity validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        toxicity_config = ToxicityConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultToxicityValidator(toxicity_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output toxicity."""
        return self._validator.validate(output)


class BiasRule(Rule):
    """Rule for validating text bias."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the rule with bias validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        bias_config = BiasConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultBiasValidator(bias_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output bias."""
        return self._validator.validate(output)


class HarmfulContentRule(Rule):
    """Rule for validating harmful content."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the rule with harmful content validation.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration dictionary
        """
        # Create config object first
        harmful_config = HarmfulContentConfig(**(config or {}))

        # Create default validator if none provided
        validator = validator or DefaultHarmfulContentValidator(harmful_config)

        # Initialize base class
        super().__init__(name=name, description=description, validator=validator)

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output harmful content."""
        return self._validator.validate(output)


def create_toxicity_rule(
    name: str = "toxicity_rule",
    description: str = "Validates text for toxic content",
    config: Optional[Dict[str, Any]] = None,
) -> ToxicityRule:
    """
    Create a toxicity rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured ToxicityRule instance
    """
    if config is None:
        config = {
            "threshold": 0.5,
            "indicators": [
                "hate",
                "offensive",
                "vulgar",
                "profanity",
                "obscene",
                "racist",
                "sexist",
                "discriminatory",
                "threatening",
                "harassing",
            ],
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return ToxicityRule(
        name=name,
        description=description,
        config=config,
    )


def create_bias_rule(
    name: str = "bias_rule",
    description: str = "Validates text for biased content",
    config: Optional[Dict[str, Any]] = None,
) -> BiasRule:
    """
    Create a bias rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured BiasRule instance
    """
    if config is None:
        config = {
            "threshold": 0.3,
            "categories": {
                "gender": [
                    "sexist",
                    "misogynistic",
                    "patriarchal",
                    "gender bias",
                    "gender stereotype",
                ],
                "racial": [
                    "racist",
                    "racial bias",
                    "ethnic stereotype",
                    "racial prejudice",
                    "discriminatory",
                ],
                "age": [
                    "ageist",
                    "age discrimination",
                    "age bias",
                    "age stereotype",
                    "generational bias",
                ],
                "cultural": [
                    "cultural bias",
                    "xenophobic",
                    "ethnocentric",
                    "cultural stereotype",
                    "cultural prejudice",
                ],
            },
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return BiasRule(
        name=name,
        description=description,
        config=config,
    )


def create_harmful_content_rule(
    name: str = "harmful_content_rule",
    description: str = "Validates text for harmful content",
    config: Optional[Dict[str, Any]] = None,
) -> HarmfulContentRule:
    """
    Create a harmful content rule with configuration.

    Args:
        name: The name of the rule
        description: Description of the rule
        config: Optional configuration dictionary

    Returns:
        Configured HarmfulContentRule instance
    """
    if config is None:
        config = {
            "categories": {
                "violence": [
                    "violent",
                    "brutal",
                    "aggressive",
                    "threatening",
                    "dangerous",
                ],
                "self_harm": [
                    "suicide",
                    "self-harm",
                    "self-injury",
                    "self-destructive",
                    "harmful behavior",
                ],
                "exploitation": [
                    "exploitative",
                    "manipulative",
                    "coercive",
                    "predatory",
                    "abusive",
                ],
                "misinformation": [
                    "false",
                    "misleading",
                    "deceptive",
                    "propaganda",
                    "disinformation",
                ],
            },
            "cache_size": 100,
            "priority": 1,
            "cost": 1.0,
        }

    return HarmfulContentRule(
        name=name,
        description=description,
        config=config,
    )


# Export public classes and functions
__all__ = [
    "ToxicityRule",
    "ToxicityConfig",
    "DefaultToxicityValidator",
    "BiasRule",
    "BiasConfig",
    "DefaultBiasValidator",
    "HarmfulContentRule",
    "HarmfulContentConfig",
    "DefaultHarmfulContentValidator",
    "create_toxicity_rule",
    "create_bias_rule",
    "create_harmful_content_rule",
]
