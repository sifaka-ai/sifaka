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
        threshold (float): Threshold for considering content toxic (0.0 to 1.0)
        indicators (List[str]): List of toxic language indicators
    """

    threshold: float = Field(default=0.5, description="Threshold for toxicity detection")
    indicators: List[str] = Field(default=[], description="List of toxic indicators")

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the toxicity rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration dictionary containing:
                   - threshold: Threshold for considering content toxic
                   - indicators: Optional list of toxic language indicators
            **kwargs: Additional arguments
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        config = config or {}
        self.threshold = config.get("threshold", 0.5)
        self.indicators = config.get("indicators", [])

    def _validate_impl(self, output: str) -> RuleResult:
        """
        Validate that the output does not contain toxic content.

        Args:
            output: The text to validate

        Returns:
            RuleResult with toxicity validation results
        """
        try:
            # Find toxic indicators in output
            found_indicators = [ind for ind in self.indicators if ind.lower() in output.lower()]

            # Calculate toxicity score based on found indicators
            toxicity_score = (
                len(found_indicators) / len(self.indicators) if self.indicators else 0.0
            )

            passed = toxicity_score <= self.threshold

            return RuleResult(
                passed=passed,
                message=(
                    f"Found toxic content with score {toxicity_score:.2f}"
                    if not passed
                    else "No toxic content detected"
                ),
                metadata={
                    "toxicity_score": toxicity_score,
                    "found_indicators": found_indicators,
                    "threshold": self.threshold,
                },
            )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during toxicity validation: {str(e)}",
                metadata={"error": str(e)},
            )


class BiasRule(Rule):
    """
    Rule that checks for biased content in the output.

    Attributes:
        categories (Dict[str, List[str]]): Dictionary of bias categories and their indicators
        threshold (float): Threshold for considering content biased (0.0 to 1.0)
    """

    categories: Dict[str, List[str]] = Field(
        default={}, description="Categories and their bias indicators"
    )
    threshold: float = Field(default=0.5, description="Threshold for bias detection")

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the bias rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration dictionary containing:
                   - categories: Dictionary of bias categories and their indicators
                   - threshold: Threshold for considering content biased
            **kwargs: Additional arguments
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        config = config or {}
        self.categories = config.get("categories", {})
        self.threshold = config.get("threshold", 0.5)

    def _validate_impl(self, output: str) -> RuleResult:
        """
        Validate that the output does not contain biased content.

        Args:
            output: The text to validate

        Returns:
            RuleResult with bias validation results
        """
        try:
            # Check each category for bias indicators
            category_scores = {}
            found_indicators = {}

            for category, indicators in self.categories.items():
                found = [ind for ind in indicators if ind.lower() in output.lower()]
                score = len(found) / len(indicators) if indicators else 0.0
                category_scores[category] = score
                if found:
                    found_indicators[category] = found

            # Overall bias score is the maximum category score
            max_score = max(category_scores.values()) if category_scores else 0.0
            passed = max_score <= self.threshold

            return RuleResult(
                passed=passed,
                message=(
                    f"Found biased content with score {max_score:.2f}"
                    if not passed
                    else "No biased content detected"
                ),
                metadata={
                    "bias_scores": category_scores,
                    "found_indicators": found_indicators,
                    "threshold": self.threshold,
                },
            )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during bias validation: {str(e)}",
                metadata={"error": str(e)},
            )


class HarmfulContentRule(Rule):
    """
    Rule that checks for harmful or dangerous content in the output.

    Attributes:
        categories (Dict[str, List[str]]): Dictionary of harmful content categories and their indicators
    """

    categories: Dict[str, List[str]] = Field(
        default={}, description="Categories of harmful content and their indicators"
    )

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the harmful content rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration dictionary containing:
                   - categories: Optional dictionary of harmful content categories
            **kwargs: Additional arguments
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        config = config or {}
        self.categories = config.get("categories", {})

    def _validate_impl(self, output: str) -> RuleResult:
        """
        Validate that the output does not contain harmful content.

        Args:
            output: The text to validate

        Returns:
            RuleResult with harmful content validation results
        """
        try:
            # Check each category for harmful content
            found_categories = []
            found_indicators = {}

            for category, indicators in self.categories.items():
                found = [ind for ind in indicators if ind.lower() in output.lower()]
                if found:
                    found_categories.append(category)
                    found_indicators[category] = found

            passed = len(found_categories) == 0

            return RuleResult(
                passed=passed,
                message=(
                    f"Found harmful content in categories: {', '.join(found_categories)}"
                    if not passed
                    else "No harmful content detected"
                ),
                metadata={
                    "found_categories": found_categories,
                    "found_indicators": found_indicators,
                },
            )
        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during harmful content validation: {str(e)}",
                metadata={"error": str(e)},
            )
