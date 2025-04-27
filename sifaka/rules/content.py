"""
Content validation rules for Sifaka.
"""

from typing import List, Dict, Any, Optional
import re
from pydantic import BaseModel, Field
from sifaka.rules.base import Rule, RuleResult
from pydantic import ConfigDict


class ProhibitedContentRule(Rule):
    """
    Rule that checks for prohibited content in the output.

    Attributes:
        prohibited_terms (List[str]): List of terms that should not appear in the output
        case_sensitive (bool): Whether the check should be case-sensitive
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prohibited_terms: List[str] = Field(
        default_factory=list, description="List of terms that should not appear in the output"
    )
    case_sensitive: bool = Field(
        default=False, description="Whether the check should be case-sensitive"
    )

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the prohibited content rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration dictionary containing:
                   - prohibited_terms: List of terms that should not appear
                   - case_sensitive: Whether to do case-sensitive matching
            **kwargs: Additional arguments

        Raises:
            ValueError: If prohibited_terms list is empty
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        config = config or {}

        # Set prohibited terms
        prohibited_terms = config.get("prohibited_terms", [])
        if not isinstance(prohibited_terms, list):
            raise ValueError("prohibited_terms must be a list")
        if not prohibited_terms:
            raise ValueError("Prohibited terms list cannot be empty")
        self.prohibited_terms = prohibited_terms

        # Set case sensitivity
        case_sensitive = config.get("case_sensitive", False)
        if not isinstance(case_sensitive, bool):
            raise ValueError("case_sensitive must be a boolean")
        self.case_sensitive = case_sensitive

    def _validate_impl(self, output: str) -> RuleResult:
        """
        Validate that the output does not contain prohibited terms.

        Args:
            output: The text to validate

        Returns:
            RuleResult with prohibited content validation results
        """
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        found_terms = []
        check_output = output if self.case_sensitive else output.lower()
        check_terms = (
            self.prohibited_terms
            if self.case_sensitive
            else [term.lower() for term in self.prohibited_terms]
        )

        for term in check_terms:
            if term in check_output:
                found_terms.append(term)

        return RuleResult(
            passed=len(found_terms) == 0,
            message=(
                "No prohibited terms found"
                if not found_terms
                else f"Found prohibited terms: {', '.join(found_terms)}"
            ),
            metadata={
                "found_terms": found_terms,
                "case_sensitive": self.case_sensitive,
            },
        )


class ToneConsistencyRule(Rule):
    """Rule that checks if the output maintains a consistent tone."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tone_indicators: Dict[str, Dict[str, List[str]]] = Field(
        default_factory=dict,
        description="Dictionary of tone categories and their positive/negative indicators",
    )
    expected_tone: str = Field(default="", description="The expected tone category")
    threshold: float = Field(
        default=0.7, description="Threshold for tone consistency", ge=0.0, le=1.0
    )

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the tone consistency rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration dictionary containing:
                   - tone_indicators: Dict of tone categories and their indicators
                   - expected_tone: The expected tone category
                   - threshold: Confidence threshold (default: 0.7)
            **kwargs: Additional arguments
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        config = config or {}

        # Set tone indicators
        tone_indicators = config.get("tone_indicators", {})
        if not isinstance(tone_indicators, dict):
            raise ValueError("tone_indicators must be a dictionary")
        for tone, indicators in tone_indicators.items():
            if not isinstance(indicators, dict) or not all(
                isinstance(v, list) for v in indicators.values()
            ):
                raise ValueError(f"Invalid indicators for tone '{tone}'. Expected dict with lists.")
        self.tone_indicators = tone_indicators

        # Set expected tone
        expected_tone = config.get("expected_tone", "")
        if not isinstance(expected_tone, str):
            raise ValueError("expected_tone must be a string")
        self.expected_tone = expected_tone

        # Set threshold
        threshold = config.get("threshold", 0.7)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a number between 0 and 1")
        self.threshold = threshold

    def _validate_impl(self, output: str) -> RuleResult:
        """
        Validate that the output maintains a consistent tone.

        Args:
            output: The text to validate

        Returns:
            RuleResult with tone consistency validation results
        """
        try:
            if not self.expected_tone:
                return RuleResult(
                    passed=True,
                    message="No expected tone specified",
                    metadata={"tone_scores": {}, "found_indicators": {}},
                )

            if self.expected_tone not in self.tone_indicators:
                return RuleResult(
                    passed=False,
                    message=f"Unknown tone category: {self.expected_tone}",
                    metadata={"available_tones": list(self.tone_indicators.keys())},
                )

            # Get indicators for expected tone
            tone_dict = self.tone_indicators[self.expected_tone]
            output_lower = output.lower()

            # Check positive and negative indicators
            found_positive = [
                ind for ind in tone_dict.get("positive", []) if ind.lower() in output_lower
            ]
            found_negative = [
                ind for ind in tone_dict.get("negative", []) if ind.lower() in output_lower
            ]

            # Calculate consistency score
            total_indicators = len(tone_dict.get("positive", [])) + len(
                tone_dict.get("negative", [])
            )
            if total_indicators == 0:
                return RuleResult(
                    passed=True,
                    message=f"No indicators defined for tone '{self.expected_tone}'",
                    metadata={"tone": self.expected_tone},
                )

            positive_ratio = (
                len(found_positive) / len(tone_dict.get("positive", []))
                if tone_dict.get("positive")
                else 0
            )
            negative_ratio = (
                len(found_negative) / len(tone_dict.get("negative", []))
                if tone_dict.get("negative")
                else 0
            )
            consistency_score = positive_ratio - negative_ratio

            passed = consistency_score >= self.threshold

            return RuleResult(
                passed=passed,
                message=(
                    f"Tone consistency score: {consistency_score:.2f}"
                    if passed
                    else f"Inconsistent tone detected (score: {consistency_score:.2f})"
                ),
                metadata={
                    "consistency_score": consistency_score,
                    "found_positive": found_positive,
                    "found_negative": found_negative,
                    "expected_tone": self.expected_tone,
                    "threshold": self.threshold,
                },
            )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during tone consistency validation: {str(e)}",
                metadata={"error": str(e)},
            )
