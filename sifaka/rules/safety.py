"""
Safety-related rules for Sifaka.
"""

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from pydantic import Field
from sifaka.rules.base import Rule, RuleResult
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToxicityConfig:
    """Configuration for toxicity validation."""

    threshold: float
    indicators: List[str]
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        if not self.indicators:
            raise ValueError("Must provide at least one toxicity indicator")
        if self.cache_size < 0:
            raise ValueError("Cache size must be non-negative")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.cost < 0:
            raise ValueError("Cost must be non-negative")


@dataclass(frozen=True)
class BiasConfig:
    """Configuration for bias validation."""

    threshold: float
    categories: Dict[str, List[str]]
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        if not self.categories:
            raise ValueError("Must provide at least one bias category")
        if self.cache_size < 0:
            raise ValueError("Cache size must be non-negative")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.cost < 0:
            raise ValueError("Cost must be non-negative")
        for category, indicators in self.categories.items():
            if not indicators:
                raise ValueError(f"Category {category} must have at least one indicator")


@dataclass(frozen=True)
class HarmfulContentConfig:
    """Configuration for harmful content validation."""

    categories: Dict[str, List[str]]
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        if not self.categories:
            raise ValueError("Must provide at least one harmful content category")
        if self.cache_size < 0:
            raise ValueError("Cache size must be non-negative")
        if self.priority < 0:
            raise ValueError("Priority must be non-negative")
        if self.cost < 0:
            raise ValueError("Cost must be non-negative")
        for category, indicators in self.categories.items():
            if not indicators:
                raise ValueError(f"Category {category} must have at least one indicator")


@runtime_checkable
class ToxicityValidator(Protocol):
    """Protocol for toxicity validation."""

    @property
    def config(self) -> ToxicityConfig:
        """Get validator configuration."""
        ...

    def validate(self, text: str) -> RuleResult:
        """Validate text for toxicity."""
        ...


@runtime_checkable
class BiasValidator(Protocol):
    """Protocol for bias validation."""

    @property
    def config(self) -> BiasConfig:
        """Get validator configuration."""
        ...

    def validate(self, text: str) -> RuleResult:
        """Validate text for bias."""
        ...


@runtime_checkable
class HarmfulContentValidator(Protocol):
    """Protocol for harmful content validation."""

    @property
    def config(self) -> HarmfulContentConfig:
        """Get validator configuration."""
        ...

    def validate(self, text: str) -> RuleResult:
        """Validate text for harmful content."""
        ...


class DefaultToxicityValidator:
    """Default implementation of toxicity validation."""

    def __init__(self, config: ToxicityConfig):
        self._config = config

    @property
    def config(self) -> ToxicityConfig:
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


class DefaultBiasValidator:
    """Default implementation of bias validation."""

    def __init__(self, config: BiasConfig):
        self._config = config

    @property
    def config(self) -> BiasConfig:
        return self._config

    def validate(self, text: str) -> RuleResult:
        """Validate text for bias."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        text_lower = text.lower()
        bias_scores: Dict[str, float] = {}

        for category, indicators in self.config.categories.items():
            found_indicators = [indicator for indicator in indicators if indicator in text_lower]
            bias_scores[category] = len(found_indicators) / len(indicators)

        overall_bias_score = sum(bias_scores.values()) / len(self.config.categories)

        if overall_bias_score > self.config.threshold:
            return RuleResult(
                passed=False,
                message=f"Output contains biased content (score: {overall_bias_score:.2f})",
                metadata={"bias_scores": bias_scores, "overall_bias_score": overall_bias_score},
            )

        return RuleResult(
            passed=True,
            message="No biased content detected",
            metadata={"bias_scores": bias_scores},
        )


class DefaultHarmfulContentValidator:
    """Default implementation of harmful content validation."""

    def __init__(self, config: HarmfulContentConfig):
        self._config = config

    @property
    def config(self) -> HarmfulContentConfig:
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
                message="Output contains potentially harmful content",
                metadata={"harmful_content": harmful_content},
            )

        return RuleResult(
            passed=True,
            message="No harmful content detected",
            metadata={"harmful_content": {}},
        )


class ToxicityRule(Rule):
    """Rule for validating content toxicity."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: ToxicityValidator,
    ) -> None:
        super().__init__(name=name, description=description)
        self._validator = validator

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output for toxicity."""
        return self._validator.validate(output)


class BiasRule(Rule):
    """Rule for validating content bias."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: BiasValidator,
    ) -> None:
        super().__init__(name=name, description=description)
        self._validator = validator

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output for bias."""
        return self._validator.validate(output)


class HarmfulContentRule(Rule):
    """Rule for validating harmful content."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: HarmfulContentValidator,
    ) -> None:
        super().__init__(name=name, description=description)
        self._validator = validator

    def _validate_impl(self, output: str) -> RuleResult:
        """Validate output for harmful content."""
        return self._validator.validate(output)


def create_toxicity_rule(
    name: str,
    description: str,
    threshold: float = 0.5,
    indicators: List[str] = None,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
) -> ToxicityRule:
    """Create a toxicity rule with default configuration."""
    if indicators is None:
        indicators = ["toxic", "hate", "offensive", "insult", "threat"]

    config = ToxicityConfig(
        threshold=threshold,
        indicators=indicators,
        cache_size=cache_size,
        priority=priority,
        cost=cost,
    )
    validator = DefaultToxicityValidator(config)
    return ToxicityRule(name=name, description=description, validator=validator)


def create_bias_rule(
    name: str,
    description: str,
    threshold: float = 0.5,
    categories: Dict[str, List[str]] = None,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
) -> BiasRule:
    """Create a bias rule with default configuration."""
    if categories is None:
        categories = {
            "gender": ["male", "female"],
            "race": ["white", "black", "asian"],
            "age": ["young", "old"],
        }

    config = BiasConfig(
        threshold=threshold,
        categories=categories,
        cache_size=cache_size,
        priority=priority,
        cost=cost,
    )
    validator = DefaultBiasValidator(config)
    return BiasRule(name=name, description=description, validator=validator)


def create_harmful_content_rule(
    name: str,
    description: str,
    categories: Dict[str, List[str]] = None,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
) -> HarmfulContentRule:
    """Create a harmful content rule with default configuration."""
    if categories is None:
        categories = {
            "violence": ["kill", "hurt", "attack", "fight"],
            "self_harm": ["suicide", "self-harm", "cutting"],
            "hate": ["hate", "racist", "sexist", "bigot"],
        }

    config = HarmfulContentConfig(
        categories=categories,
        cache_size=cache_size,
        priority=priority,
        cost=cost,
    )
    validator = DefaultHarmfulContentValidator(config)
    return HarmfulContentRule(name=name, description=description, validator=validator)
