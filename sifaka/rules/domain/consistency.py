"""
Consistency validation rules for Sifaka.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from sifaka.rules.base import Rule, RuleConfig, RuleResult, RuleValidator
from sifaka.rules.domain.base import BaseDomainValidator


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


class DefaultConsistencyValidator(BaseDomainValidator):
    """Default implementation of consistency validation."""

    def __init__(self, config: ConsistencyConfig) -> None:
        """Initialize with configuration."""
        super().__init__(config)
        self._consistency_patterns = {
            k: re.compile(pattern) for k, pattern in config.consistency_patterns.items()
        }
        self._contradiction_indicators = [
            (re.compile(pos), re.compile(neg)) for pos, neg in config.contradiction_indicators
        ]

    @property
    def config(self) -> ConsistencyConfig:
        """Get the validator configuration."""
        return self._config

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate consistency in text."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        try:
            # Check for consistency patterns
            consistency_matches = {}
            for name, pattern in self._consistency_patterns.items():
                consistency_matches[name] = len(pattern.findall(text))

            # Check for contradictions
            contradictions = []
            for pos_pattern, neg_pattern in self._contradiction_indicators:
                pos_matches = pos_pattern.findall(text)
                neg_matches = neg_pattern.findall(text)
                if pos_matches and neg_matches:
                    contradictions.append(
                        {
                            "positive": pos_matches,
                            "negative": neg_matches,
                            "pattern": (pos_pattern.pattern, neg_pattern.pattern),
                        }
                    )

            # Check for excessive repetition
            words = re.findall(r"\b\w+\b", text.lower())
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Only check for repetition of meaningful words
                    word_counts[word] = word_counts.get(word, 0) + 1

            total_words = len(words)
            repeated_words = {
                word: count
                for word, count in word_counts.items()
                if count > 1 and count / total_words > self.config.repetition_threshold
            }

            # Determine overall consistency
            passed = not contradictions and not repeated_words
            message = "Consistency validation "
            if passed:
                message += "passed"
            else:
                if contradictions:
                    message += f"failed: found {len(contradictions)} contradictions"
                elif repeated_words:
                    message += "failed: excessive word repetition detected"

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
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error validating consistency: {str(e)}",
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
    ) -> None:
        """
        Initialize the consistency rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            validator: Optional custom validator implementation
            config: Optional configuration
        """
        # Store parameters for creating the default validator
        self._rule_params = {}
        if config:
            # For backward compatibility, check both params and metadata
            params_source = config.params if config.params else config.metadata
            self._rule_params = params_source

        # Initialize base class
        super().__init__(name=name, description=description, config=config, validator=validator)

    def _create_default_validator(self) -> DefaultConsistencyValidator:
        """Create a default validator from config."""
        consistency_config = ConsistencyConfig(**self._rule_params)
        return DefaultConsistencyValidator(consistency_config)


def create_consistency_rule(
    name: str = "consistency_rule",
    description: str = "Validates content consistency",
    config: Optional[Dict[str, Any]] = None,
) -> ConsistencyRule:
    """
    Create a consistency validation rule.

    Args:
        name: Name of the rule
        description: Description of the rule
        config: Optional configuration

    Returns:
        A configured ConsistencyRule
    """
    # Convert the dictionary config to RuleConfig with params
    rule_config = RuleConfig(params=config or {})

    return ConsistencyRule(
        name=name,
        description=description,
        config=rule_config,
    )
