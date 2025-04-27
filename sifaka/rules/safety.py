"""
Safety-related rules for Sifaka.
"""

from typing import Dict, Any, List, Optional
from pydantic import Field
from sifaka.rules.base import Rule, RuleResult


class ToxicityRule(Rule):
    """
    Rule that checks for toxic content in the output.

    Attributes:
        toxicity_threshold (float): Threshold for considering content toxic (0.0 to 1.0)
        toxicity_indicators (List[str]): List of toxic language indicators
    """

    toxicity_threshold: float = Field(default=0.5)
    toxicity_indicators: List[str] = Field(
        default=[
            "hate",
            "racist",
            "sexist",
            "bigot",
            "nazi",
            "kill",
            "murder",
            "suicide",
            "harm",
            "dangerous",
            "illegal",
            "criminal",
            "terrorist",
        ]
    )

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output does not contain toxic content.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation
        """
        output_lower = output.lower()
        toxic_indicators = []

        for indicator in self.toxicity_indicators:
            if indicator in output_lower:
                toxic_indicators.append(indicator)

        toxicity_score = len(toxic_indicators) / len(self.toxicity_indicators)

        if toxicity_score > self.toxicity_threshold:
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


class BiasRule(Rule):
    """
    Rule that checks for biased content in the output.

    Attributes:
        bias_indicators (Dict[str, List[str]]): Dictionary of bias categories and their indicators
        bias_threshold (float): Threshold for considering content biased (0.0 to 1.0)
    """

    bias_indicators: Dict[str, List[str]] = Field(
        default={
            "gender": ["he", "she", "man", "woman", "male", "female"],
            "race": ["white", "black", "asian", "hispanic", "native"],
            "age": ["young", "old", "elderly", "youth"],
            "religion": ["christian", "muslim", "jewish", "hindu", "buddhist"],
            "political": ["liberal", "conservative", "democrat", "republican"],
        }
    )
    bias_threshold: float = Field(default=0.3)

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output does not contain biased content.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation
        """
        output_lower = output.lower()
        bias_scores = {}
        total_indicators = 0

        for category, indicators in self.bias_indicators.items():
            found_indicators = []
            for indicator in indicators:
                if indicator in output_lower:
                    found_indicators.append(indicator)
            bias_scores[category] = len(found_indicators) / len(indicators)
            total_indicators += len(indicators)

        overall_bias_score = sum(bias_scores.values()) / len(self.bias_indicators)

        if overall_bias_score > self.bias_threshold:
            return RuleResult(
                passed=False,
                message=f"Output contains biased content (score: {overall_bias_score:.2f})",
                metadata={"bias_scores": bias_scores, "overall_bias_score": overall_bias_score},
            )

        return RuleResult(
            passed=True, message="No biased content detected", metadata={"bias_scores": bias_scores}
        )


class HarmfulContentRule(Rule):
    """
    Rule that checks for harmful or dangerous content in the output.

    Attributes:
        harmful_categories (Dict[str, List[str]]): Dictionary of harmful content categories and their indicators
    """

    harmful_categories: Dict[str, List[str]] = Field(
        default={
            "violence": ["kill", "murder", "attack", "weapon", "gun", "bomb"],
            "self_harm": ["suicide", "self-harm", "cutting", "overdose"],
            "illegal": ["drugs", "illegal", "criminal", "fraud", "hack"],
            "medical": ["cure", "treatment", "diagnosis", "prescription"],
            "financial": ["investment", "stock", "crypto", "bitcoin", "scam"],
        }
    )

    class Config:
        arbitrary_types_allowed = True

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output does not contain harmful content.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation
        """
        output_lower = output.lower()
        harmful_content = {}

        for category, indicators in self.harmful_categories.items():
            found_indicators = []
            for indicator in indicators:
                if indicator in output_lower:
                    found_indicators.append(indicator)
            if found_indicators:
                harmful_content[category] = found_indicators

        if harmful_content:
            return RuleResult(
                passed=False,
                message="Output contains potentially harmful content",
                metadata={"harmful_content": harmful_content},
            )

        return RuleResult(passed=True, message="No harmful content detected")
