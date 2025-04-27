"""
Fact-checking rules for Sifaka.
"""

from typing import Dict, Any, List, Optional, Set
from pydantic import Field
from sifaka.rules.base import Rule, RuleResult
import re


class FactualConsistencyRule(Rule):
    """
    Rule that checks for factual consistency within the output.

    Attributes:
        contradiction_indicators (List[str]): List of phrases that indicate contradictions
        confidence_threshold (float): Threshold for considering a statement confident (0.0 to 1.0)
    """

    contradiction_indicators: List[str] = Field(
        default=[
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
    confidence_threshold: float = Field(default=0.7)

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output maintains factual consistency.

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

        output_lower = output.lower()
        contradictions = []

        for indicator in self.contradiction_indicators:
            if indicator in output_lower:
                contradictions.append(indicator)

        if contradictions:
            return RuleResult(
                passed=False,
                message="Output contains potential contradictions",
                metadata={"contradiction_indicators": contradictions},
            )

        return RuleResult(passed=True, message="No contradictions detected")


class ConfidenceRule(Rule):
    """
    Rule that checks for appropriate confidence levels in statements.

    Attributes:
        confidence_indicators (Dict[str, List[str]]): Dictionary of confidence levels and their indicators
    """

    confidence_indicators: Dict[str, List[str]] = Field(
        default={
            "high": ["definitely", "certainly", "always", "never", "must", "will"],
            "medium": ["likely", "probably", "usually", "often", "generally"],
            "low": ["maybe", "possibly", "sometimes", "occasionally", "might"],
            "uncertain": ["perhaps", "could", "may", "seems", "appears"],
        }
    )

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output uses appropriate confidence levels.

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

        output_lower = output.lower()
        confidence_levels = {}

        for level, indicators in self.confidence_indicators.items():
            found_indicators = []
            for indicator in indicators:
                if indicator in output_lower:
                    found_indicators.append(indicator)
            if found_indicators:
                confidence_levels[level] = found_indicators

        if confidence_levels:
            return RuleResult(
                passed=True,
                message="Confidence levels detected",
                metadata={"confidence_levels": confidence_levels},
            )

        return RuleResult(passed=True, message="No confidence indicators detected")


class CitationRule(Rule):
    """
    Rule that checks for proper citations and references.

    Attributes:
        citation_patterns (List[str]): List of regex patterns for citations
        required_citations (bool): Whether citations are required
    """

    citation_patterns: List[str] = Field(
        default=[
            r"\[[\d]+\]",  # [1], [2], etc.
            r"\([A-Za-z]+ et al., \d{4}\)",  # (Smith et al., 2020)
            r"\([A-Za-z]+, \d{4}\)",  # (Smith, 2020)
            r"https?://[^\s]+",  # URLs
        ]
    )
    required_citations: bool = Field(default=True)

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output contains proper citations.

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

        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, output)
            citations.extend(matches)

        if self.required_citations and not citations:
            return RuleResult(
                passed=False,
                message="No citations found in the output",
                metadata={"citation_patterns": self.citation_patterns},
            )

        return RuleResult(
            passed=True,
            message=f"Found {len(citations)} citations",
            metadata={"citations": citations},
        )


class FactualAccuracyRule(Rule):
    """
    Rule that checks for factual accuracy using a knowledge base.

    Attributes:
        knowledge_base (Dict[str, Set[str]]): Dictionary of facts and their variations
    """

    knowledge_base: Dict[str, Set[str]] = Field(
        default={
            "earth_shape": {"round", "spherical", "geoid"},
            "gravity": {"9.8 m/s²", "9.8 meters per second squared"},
            "pi": {"3.14159", "3.14", "π"},
            "water_boiling_point": {"100°C", "212°F", "373.15K"},
        }
    )

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output contains factually accurate information.

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

        output_lower = output.lower()
        inaccuracies = []

        for fact, variations in self.knowledge_base.items():
            found = False
            for variation in variations:
                if variation.lower() in output_lower:
                    found = True
                    break
            if not found:
                inaccuracies.append(fact)

        if inaccuracies:
            return RuleResult(
                passed=False,
                message="Output contains potentially inaccurate information",
                metadata={"inaccuracies": inaccuracies},
            )

        return RuleResult(passed=True, message="No factual inaccuracies detected")
