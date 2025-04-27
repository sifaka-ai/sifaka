from typing import Optional, Dict, Any
from sifaka.rules.base import Rule, RuleResult


class ToxicityRule(Rule):
    """
    Rule that checks if the output contains toxic content.

    This rule is part of the Sifaka validation framework and implements toxicity
    analysis for text content. It uses a toxicity detection model to determine
    if the output contains harmful or inappropriate content.

    Architecture Notes:
    - Inherits from the base Rule class to implement the validation contract
    - Uses a toxicity detection model to score text
    - Returns RuleResult objects containing validation status, messages, and metadata
    - Follows the single responsibility principle by focusing only on toxicity validation
    - Includes error handling for toxicity analysis failures

    Data Flow:
    1. User creates ToxicityRule with desired maximum toxicity threshold
    2. validate() method receives output text
    3. Toxicity analysis is performed on the text
    4. Score is compared against threshold
    5. Result is wrapped in RuleResult with relevant metadata
    6. RuleResult is returned to the caller

    Usage Example:
        rule = ToxicityRule(
            name="toxicity_rule",
            description="Checks for toxic content",
            config={"max_toxicity": 0.5}
        )
    """

    max_toxicity: float = 0.5

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the rule with a maximum toxicity threshold.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration dictionary containing:
                   - max_toxicity: Maximum allowed toxicity score (inclusive).
                                 Must be between 0.0 (no toxicity) and 1.0 (highly toxic).
                                 Defaults to 0.5 (moderate tolerance).
            **kwargs: Additional arguments

        Raises:
            ValueError: If max_toxicity is not between 0.0 and 1.0
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        # Extract toxicity threshold from config
        config = config or {}
        max_toxicity = config.get("max_toxicity", 0.5)

        if not 0.0 <= max_toxicity <= 1.0:
            raise ValueError("max_toxicity must be between 0.0 and 1.0")

        # Set the value using object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "max_toxicity", max_toxicity)

    def validate(self, output: str) -> RuleResult:
        """
        Validate that the output's toxicity is below the maximum threshold.

        This method implements the core validation logic by:
        1. Analyzing the toxicity of the output text
        2. Comparing the toxicity score against the threshold
        3. Constructing a detailed result message
        4. Packaging the result with relevant metadata

        Args:
            output: The text to validate

        Returns:
            RuleResult: Contains:
                       - passed: Boolean indicating if toxicity is below threshold
                       - message: Human-readable validation result
                       - metadata: Additional validation details including scores
        """
        try:
            # TODO: Replace with actual toxicity analysis implementation
            toxicity_score = 0.0  # Placeholder implementation
            passed = toxicity_score <= self.max_toxicity

            return RuleResult(
                passed=passed,
                message=(
                    f"Output toxicity {toxicity_score:.2f} "
                    f"{'is below' if passed else 'exceeds'} "
                    f"maximum threshold of {self.max_toxicity}"
                ),
                metadata={"toxicity_score": toxicity_score, "max_toxicity": self.max_toxicity},
            )

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during toxicity analysis: {str(e)}",
                metadata={"error": str(e), "max_toxicity": self.max_toxicity},
            )
