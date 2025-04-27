"""
Content validation rules for Sifaka.
"""

from typing import List, Dict, Any, Optional
import re
from pydantic import BaseModel, Field
from sifaka.rules.base import Rule, RuleResult


class ProhibitedContentRule(Rule):
    """
    Rule that checks for prohibited content in the output.

    Attributes:
        prohibited_terms (List[str]): List of terms that should not appear in the output
        case_sensitive (bool): Whether the check should be case-sensitive
    """

    prohibited_terms: List[str] = []
    case_sensitive: bool = False

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output does not contain prohibited terms.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If output is None
        """
        if output is None:
            raise ValueError("Output cannot be None")

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
                "Prohibited content detected" if found_terms else "No prohibited content detected"
            ),
            metadata={
                "found_terms": found_terms if found_terms else None,
                "case_sensitive": self.case_sensitive,
            },
        )


class ToneConsistencyRule(Rule):
    """
    Rule that checks for consistency in tone throughout the output.

    Attributes:
        expected_tone (str): The expected tone of the output (formal, informal, etc.)
        tone_indicators (Dict[str, Dict[str, List[str]]]): Dictionary of tone indicators
    """

    expected_tone: str
    tone_indicators: Dict[str, Dict[str, List[str]]] = Field(
        default={
            "formal": {
                "positive": ["therefore", "consequently", "furthermore", "thus", "hence"],
                "negative": ["yeah", "cool", "awesome", "btw", "gonna", "wanna"],
            },
            "informal": {
                "positive": ["yeah", "cool", "awesome", "btw", "gonna", "wanna"],
                "negative": ["therefore", "consequently", "furthermore", "thus", "hence"],
            },
        }
    )

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output maintains the expected tone.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If output is None
        """
        if output is None:
            raise ValueError("Output cannot be None")

        if self.expected_tone.lower() not in self.tone_indicators:
            return RuleResult(
                passed=False,
                message=f"Unknown tone: {self.expected_tone}",
                metadata={"available_tones": list(self.tone_indicators.keys())},
            )

        output_lower = output.lower()

        # Check for positive indicators
        positive_indicators = []
        for term in self.tone_indicators[self.expected_tone.lower()]["positive"]:
            if term in output_lower:
                positive_indicators.append(term)

        # Check for negative indicators
        negative_indicators = []
        for term in self.tone_indicators[self.expected_tone.lower()]["negative"]:
            if term in output_lower:
                negative_indicators.append(term)

        # Simple scoring (in a real implementation, this would be more sophisticated)
        if len(negative_indicators) > len(positive_indicators):
            return RuleResult(
                passed=False,
                message=f"Output does not maintain {self.expected_tone} tone",
                metadata={
                    "positive_indicators": positive_indicators,
                    "negative_indicators": negative_indicators,
                },
            )

        return RuleResult(
            passed=True,
            message=f"Output maintains {self.expected_tone} tone",
            metadata={
                "positive_indicators": positive_indicators,
                "negative_indicators": negative_indicators,
            },
        )
