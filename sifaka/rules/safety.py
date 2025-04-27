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

    toxicity_threshold: float = 0.5
    toxicity_indicators: List[str] = [
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
                   - toxicity_threshold: Threshold for considering content toxic
                   - toxicity_indicators: Optional list of toxic language indicators
            **kwargs: Additional arguments
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        # Extract configuration
        config = config or {}
        if "toxicity_threshold" in config:
            self.toxicity_threshold = config["toxicity_threshold"]
        if "toxicity_indicators" in config:
            self.toxicity_indicators = config["toxicity_indicators"]

    def validate(self, output: str) -> RuleResult:
        """
        Validate that the output does not contain toxic content.

        Args:
            output (str): The LLM output to validate

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If output is None or not a string
        """
        if output is None:
            raise ValueError("Output cannot be None")
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        found_indicators = [
            indicator
            for indicator in self.toxicity_indicators
            if indicator.lower() in output.lower()
        ]
        toxicity_score = len(found_indicators) / max(1, len(found_indicators))

        metadata = {
            "toxicity_score": toxicity_score,
            "toxicity_threshold": self.toxicity_threshold,
            "toxic_indicators": found_indicators,
        }

        passed = toxicity_score <= self.toxicity_threshold
        message = "No toxic content detected." if passed else "Toxic content detected."

        return RuleResult(passed=passed, message=message, metadata=metadata)


class BiasRule(Rule):
    """
    Rule that checks for biased content in the output.

    Attributes:
        bias_indicators (Dict[str, List[str]]): Dictionary of bias categories and their indicators
        bias_threshold (float): Threshold for considering content biased (0.0 to 1.0)
    """

    bias_threshold: float = 0.3
    bias_indicators: Dict[str, List[str]] = {
        "gender": ["male", "female", "man", "woman", "boy", "girl"],
        "race": ["white", "black", "asian", "hispanic", "latino", "arab"],
        "age": ["young", "old", "elderly", "senior", "teen", "adult"],
        "religion": ["christian", "muslim", "jewish", "hindu", "buddhist"],
        "political": ["liberal", "conservative", "democrat", "republican"],
    }

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
                   - bias_threshold: Threshold for considering content biased
                   - bias_indicators: Optional dictionary of bias categories and indicators
            **kwargs: Additional arguments
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        # Extract configuration
        config = config or {}
        if "bias_threshold" in config:
            self.bias_threshold = config["bias_threshold"]
        if "bias_indicators" in config:
            self.bias_indicators = config["bias_indicators"]

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output does not contain biased content.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If output is None or not a string
        """
        if output is None:
            raise ValueError("Output cannot be None")
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        output_lower = output.lower()
        bias_scores = {}
        found_indicators = {}

        for category, indicators in self.bias_indicators.items():
            found = []
            for indicator in indicators:
                if indicator in output_lower:
                    found.append(indicator)
            if found:
                # Calculate bias score based on found indicators only
                bias_scores[category] = len(found) / max(1, len(found))
                found_indicators[category] = found
            else:
                bias_scores[category] = 0.0

        max_bias_score = max(bias_scores.values()) if bias_scores else 0.0

        return RuleResult(
            passed=max_bias_score < self.bias_threshold,
            message=(
                "Biased content detected"
                if max_bias_score >= self.bias_threshold
                else "No biased content detected"
            ),
            metadata={
                "bias_scores": bias_scores,
                "bias_threshold": self.bias_threshold,
                "found_indicators": found_indicators,
            },
        )


class HarmfulContentRule(Rule):
    """
    Rule that checks for harmful or dangerous content in the output.

    Attributes:
        harmful_categories (Dict[str, List[str]]): Dictionary of harmful content categories and their indicators
    """

    harmful_categories: Dict[str, List[str]] = {
        "violence": ["weapon", "gun", "bomb", "attack", "fight", "assault"],
        "self_harm": ["suicide", "self-harm", "cutting", "overdose", "depression"],
        "hate_speech": ["hate", "racist", "sexist", "homophobic", "discrimination"],
        "medical": ["treatment", "medicine", "diagnosis", "disease"],
        "financial": ["invest", "money", "stock", "crypto", "loan", "debt"],
        "illegal": ["fraud", "hack", "steal", "theft", "crime", "criminal", "illegal", "drugs"],
    }

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
                   - harmful_categories: Optional dictionary of harmful content categories
            **kwargs: Additional arguments
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        # Extract configuration
        config = config or {}
        if "harmful_categories" in config:
            self.harmful_categories = config["harmful_categories"]

    def validate(self, output: str) -> RuleResult:
        """
        Validate that the output does not contain harmful content.

        Args:
            output (str): The LLM output to validate

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If output is None or not a string
        """
        if output is None:
            raise ValueError("Output cannot be None")
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        found_categories = []
        for category, indicators in self.harmful_categories.items():
            if any(indicator.lower() in output.lower() for indicator in indicators):
                found_categories.append(category)

        metadata = {}
        if found_categories:
            metadata["harmful_content"] = found_categories

        passed = len(found_categories) == 0
        message = "No harmful content detected." if passed else "Harmful content detected."

        return RuleResult(passed=passed, message=message, metadata=metadata)
