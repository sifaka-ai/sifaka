"""
Consistency validation rules for Sifaka.

This module provides validators and rules for checking consistency in text.

Configuration Pattern:
    This module follows the standard Sifaka configuration pattern:
    - All rule-specific configuration is stored in RuleConfig.params
    - The ConsistencyConfig class extends RuleConfig and provides type-safe access to parameters
    - Factory functions (create_consistency_rule, create_consistency_validator) handle configuration

Usage Example:
    from sifaka.rules.domain.consistency import create_consistency_rule

    # Create a consistency rule using the factory function
    rule = create_consistency_rule(
        consistency_patterns={
            "present": r"\\b(?:is|are|am)\\b",
            "past": r"\\b(?:was|were)\\b"
        },
        repetition_threshold=0.2
    )

    # Validate text
    result = rule.validate("This text is consistent and was written carefully.")

    # Alternative: Create with explicit RuleConfig
    from sifaka.rules.base import BaseValidator, RuleConfig, Any
    rule = ConsistencyRule(
        config=RuleConfig(
            params={
                "consistency_patterns": {
                    "present": r"\\b(?:is|are|am)\\b",
                    "past": r"\\b(?:was|were)\\b"
                },
                "repetition_threshold": 0.2
            }
        )
    )
"""

# Standard library
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

# Third-party
from pydantic import BaseModel, Field, PrivateAttr

# Sifaka
from sifaka.rules.base import Rule, RuleConfig, RuleResult, RuleValidator
from sifaka.rules.domain.base import BaseDomainValidator


__all__ = [
    # Config classes
    "ConsistencyConfig",
    # Protocol classes
    "ConsistencyValidator",
    # Validator classes
    "DefaultConsistencyValidator",
    # Rule classes
    "ConsistencyRule",
    # Factory functions
    "create_consistency_validator",
    "create_consistency_rule",
    # Internal helpers (not exported but left here for discoverability)
    "_PatternAnalyzer",
    "_ContradictionAnalyzer",
    "_RepetitionAnalyzer",
]


@dataclass(frozen=True)
class ConsistencyConfig(RuleConfig):
    """Configuration for consistency rules."""

    consistency_patterns: Dict[str, str] = field(
        default_factory=lambda: {
            "present": r"\b(?:is|are|am|has|have|do|does)\b",
            "past": r"\b(?:was|were|had|did)\b",
            "future": r"\b(?:will|shall|going to)\b",
            "first_person": r"\b(?:I|we|my|our|myself|ourselves)\b",
            "second_person": r"\b(?:you|your|yourself|yourselves)\b",
            "third_person": r"\b(?:he|she|it|they|his|her|its|their|himself|herself|itself|themselves)\b",
            "active": r"\b(?:subject)\s+(?:verb)\b",
            "passive": r"\b(?:is|are|was|were)\s+(?:\w+ed|\w+en)\b",
            "list_marker": r"(?m)^[-*•]\s+|\d+\.\s+",
            "code_block": r"```[\s\S]*?```|`[^`]+`",
            "table_marker": r"\|[^|]+\|",
            "heading": r"(?m)^#{1,6}\s+\w+",
        }
    )
    contradiction_indicators: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            (r"\b(?:is|are)\b", r"\b(?:is not|are not|isn't|aren't)\b"),
            (r"\b(?:will|shall)\b", r"\b(?:will not|shall not|won't|shan't)\b"),
            (r"\b(?:must|should)\b", r"\b(?:must not|should not|shouldn't)\b"),
            (r"\b(?:always|never)\b", r"\b(?:sometimes|occasionally)\b"),
            (r"\b(?:all|every)\b", r"\b(?:some|few|none)\b"),
            (r"\b(?:increase|rise)\b", r"\b(?:decrease|fall)\b"),
            (r"\b(?:more|greater)\b", r"\b(?:less|fewer)\b"),
            (r"\b(?:begin|start)\b", r"\b(?:end|finish)\b"),
        ]
    )
    repetition_threshold: float = 0.3
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        super().__post_init__()
        if not 0.0 <= self.repetition_threshold <= 1.0:
            raise ValueError("repetition_threshold must be between 0.0 and 1.0")
        if not self.consistency_patterns:
            raise ValueError("Must provide at least one consistency pattern")
        if not self.contradiction_indicators:
            raise ValueError("Must provide at least one contradiction indicator")


@runtime_checkable
class ConsistencyValidator(Protocol):
    """Protocol for consistency validation."""

    def validate(self, text: str) -> RuleResult: ...
    @property
    def config(self) -> ConsistencyConfig: ...


class _PatternAnalyzer(BaseModel):
    """Analyze occurrences of configured regex patterns in a text."""

    patterns: Dict[str, str] = Field(default_factory=dict)

    # Compiled regex patterns stored privately to avoid re-compilation on each call
    _compiled: Dict[str, re.Pattern] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        # Compile provided regex patterns for performance
        self._compiled = {name: re.compile(expr) for name, expr in self.patterns.items()}

    # Public API -----------------------------------------------------------
    def analyze(self, text: str) -> Dict[str, int]:
        """Return a mapping of pattern name to number of matches found in *text*."""

        return {name: len(regex.findall(text)) for name, regex in self._compiled.items()}


class _ContradictionAnalyzer(BaseModel):
    """Detect positive/negative pattern contradictions in a text."""

    indicators: List[Tuple[str, str]] = Field(default_factory=list)

    _compiled: List[Tuple[re.Pattern, re.Pattern]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._compiled = [(re.compile(pos), re.compile(neg)) for pos, neg in self.indicators]

    # Public API -----------------------------------------------------------
    def analyze(self, text: str) -> List[Dict[str, Any]]:  # noqa: ANN401 – dynamic structure
        """Return a list of detected contradictions with match details."""

        contradictions: List[Dict[str, Any]] = []
        for pos_regex, neg_regex in self._compiled:
            pos_matches = pos_regex.findall(text)
            neg_matches = neg_regex.findall(text)
            if pos_matches and neg_matches:
                contradictions.append(
                    {
                        "positive": pos_matches,
                        "negative": neg_matches,
                        "pattern": (pos_regex.pattern, neg_regex.pattern),
                    }
                )
        return contradictions


class _RepetitionAnalyzer(BaseModel):
    """Identify words whose repetition rate exceeds a threshold."""

    threshold: float = Field(gt=0.0, le=1.0, default=0.3)

    # Public API -----------------------------------------------------------
    def analyze(self, text: str) -> Dict[str, int]:
        """Return a mapping of overly-repeated words to their counts."""

        words = re.findall(r"\b\w+\b", text.lower())
        total_words = len(words)
        if total_words == 0:
            return {}

        counts: Counter[str] = Counter(
            w for w in words if len(w) > 3  # focus on meaningful words
        )

        return {
            word: count
            for word, count in counts.items()
            if count > 1 and count / total_words > self.threshold
        }


class DefaultConsistencyValidator(BaseDomainValidator):
    """Default implementation of consistency validation."""

    def __init__(self, config: ConsistencyConfig) -> None:
        """Initialize validator with *config* and dedicated analyzer components."""

        super().__init__(config)

        # Compose responsibilities into dedicated, testable analyzer objects
        self._pattern_analyzer = _PatternAnalyzer(patterns=config.consistency_patterns)
        self._contradiction_analyzer = _ContradictionAnalyzer(
            indicators=config.contradiction_indicators
        )
        self._repetition_analyzer = _RepetitionAnalyzer(
            threshold=config.repetition_threshold
        )

    @property
    def config(self) -> ConsistencyConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate consistency in text."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        try:
            # ------------------------------------------------------------------
            # Delegate to analyzers
            # ------------------------------------------------------------------
            consistency_matches = self._pattern_analyzer.analyze(text)
            contradictions = self._contradiction_analyzer.analyze(text)
            repeated_words = self._repetition_analyzer.analyze(text)

            # ------------------------------------------------------------------
            # Compose final result
            # ------------------------------------------------------------------
            passed = not contradictions and not repeated_words

            message = (
                "Consistency validation passed"
                if passed
                else (
                    f"Consistency validation failed: found {len(contradictions)} contradictions"
                    if contradictions
                    else "Consistency validation failed: excessive word repetition detected"
                )
            )

            return RuleResult(
                passed=passed,
                message=message,
                metadata={
                    "consistency_matches": consistency_matches,
                    "contradictions": contradictions,
                    "repeated_words": repeated_words,
                    "repetition_threshold": self.config.repetition_threshold,
                },
            )
        except Exception as e:  # pragma: no cover – safeguard
            return RuleResult(
                passed=False,
                message=f"Error validating consistency: {e}",
                metadata={"error": str(e)},
            )


class ConsistencyRule(Rule):
    """Rule that checks for consistency in text."""

    def __init__(
        self,
        name: str = "consistency_rule",
        description: str = "Checks for consistency in text",
        validator: Optional[RuleValidator[str]] = None,
        config: Optional[RuleConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the consistency rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration
            **kwargs: Additional keyword arguments for the rule
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config and config.params:
            self._rule_params = config.params

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            validator=validator,
            **kwargs,
        )

    def _create_default_validator(self) -> DefaultConsistencyValidator:
        """Create a default validator from config."""
        consistency_config = ConsistencyConfig(**self._rule_params)
        return DefaultConsistencyValidator(consistency_config)


def create_consistency_validator(
    consistency_patterns: Optional[Dict[str, str]] = None,
    contradiction_indicators: Optional[List[Tuple[str, str]]] = None,
    repetition_threshold: Optional[float] = None,
    **kwargs,
) -> DefaultConsistencyValidator:
    """
    Create a consistency validator with the specified configuration.

    This factory function creates a configured consistency validator instance.
    It's useful when you need a validator without creating a full rule.

    Args:
        consistency_patterns: Dictionary of regex patterns for consistency checks
        contradiction_indicators: List of tuples with positive and negative pattern pairs
        repetition_threshold: Threshold for detecting excessive repetition (0.0 to 1.0)
        **kwargs: Additional keyword arguments for the config

    Returns:
        Configured consistency validator
    """
    # Extract RuleConfig parameters from kwargs
    rule_config_params = {}
    for param in ["priority", "cache_size", "cost", "params"]:
        if param in kwargs:
            rule_config_params[param] = kwargs.pop(param)

    # Create config with default or provided values
    config_params = {}
    if consistency_patterns is not None:
        config_params["consistency_patterns"] = consistency_patterns
    if contradiction_indicators is not None:
        config_params["contradiction_indicators"] = contradiction_indicators
    if repetition_threshold is not None:
        config_params["repetition_threshold"] = repetition_threshold

    # Add any remaining config parameters
    config_params.update(rule_config_params)

    # Create the config
    config = ConsistencyConfig(**config_params)

    # Return configured validator
    return DefaultConsistencyValidator(config)


def create_consistency_rule(
    name: str = "consistency_rule",
    description: str = "Validates content consistency",
    consistency_patterns: Optional[Dict[str, str]] = None,
    contradiction_indicators: Optional[List[Tuple[str, str]]] = None,
    repetition_threshold: Optional[float] = None,
    **kwargs,
) -> ConsistencyRule:
    """
    Create a consistency validation rule.

    This factory function creates a configured ConsistencyRule instance.
    It uses create_consistency_validator internally to create the validator.

    Args:
        name: Name of the rule
        description: Description of the rule
        consistency_patterns: Dictionary of regex patterns for consistency checks
        contradiction_indicators: List of tuples with positive and negative pattern pairs
        repetition_threshold: Threshold for detecting excessive repetition (0.0 to 1.0)
        **kwargs: Additional keyword arguments for the rule

    Returns:
        A configured ConsistencyRule
    """
    # Create validator using the validator factory
    validator = create_consistency_validator(
        consistency_patterns=consistency_patterns,
        contradiction_indicators=contradiction_indicators,
        repetition_threshold=repetition_threshold,
        **{k: v for k, v in kwargs.items() if k in ["priority", "cache_size", "cost", "params"]},
    )

    # Extract rule-specific kwargs
    rule_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["priority", "cache_size", "cost", "params"]
    }

    # Create and return rule
    return ConsistencyRule(
        name=name,
        description=description,
        validator=validator,
        **rule_kwargs,
    )
