from typing import List, Optional, Dict, Any
from pydantic import Field
from sifaka.rules.base import Rule, RuleResult


class ProhibitedContentRule(Rule):
    """
    Rule that checks if the output contains any prohibited terms.

    This rule is part of the Sifaka validation framework and implements content
    filtering by checking for prohibited terms in the output text.

    Architecture Notes:
    - Inherits from the base Rule class to implement the validation contract
    - Uses case-insensitive matching for prohibited terms
    - Returns RuleResult objects containing validation status, messages, and metadata
    - Follows the single responsibility principle by focusing only on content filtering
    - Includes error handling for validation failures

    Data Flow:
    1. User creates ProhibitedContentRule with list of prohibited terms
    2. validate() method receives output text
    3. Text is checked for prohibited terms
    4. Result is wrapped in RuleResult with relevant metadata
    5. RuleResult is returned to the caller

    Usage Example:
        rule = ProhibitedContentRule(
            name="content_rule",
            description="Checks for prohibited terms",
            config={"prohibited_terms": ["bad", "inappropriate", "forbidden"]}
        )
        result = rule.validate("This is a bad example")
        # result.passed will be False
        # result.message will indicate which terms were found
    """

    prohibited_terms: List[str] = Field(
        default=[], description="List of terms that should not appear in the output"
    )

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the rule with a list of prohibited terms.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration dictionary containing:
                   - prohibited_terms: List of terms that should not appear in the output
            **kwargs: Additional arguments

        Raises:
            ValueError: If prohibited_terms list is empty
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        # Extract prohibited terms from config
        config = config or {}
        prohibited_terms = config.get("prohibited_terms", [])

        if not prohibited_terms:
            raise ValueError("prohibited_terms list cannot be empty")

        # Set the values using object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "prohibited_terms", prohibited_terms)

    def validate(self, output: str) -> RuleResult:
        """
        Validate that the output does not contain any prohibited terms.

        This method implements the core validation logic by:
        1. Converting output to lowercase for case-insensitive matching
        2. Checking for each prohibited term
        3. Constructing a detailed result message
        4. Packaging the result with relevant metadata

        Args:
            output: The text to validate

        Returns:
            RuleResult: Contains:
                       - passed: Boolean indicating if no prohibited terms found
                       - message: Human-readable validation result
                       - metadata: Additional validation details including found terms
        """
        try:
            output_lower = output.lower()
            found_terms = [term for term in self.prohibited_terms if term.lower() in output_lower]
            passed = not found_terms

            return RuleResult(
                passed=passed,
                message=(
                    f"Found prohibited terms: {', '.join(found_terms)}"
                    if found_terms
                    else "No prohibited terms found"
                ),
                metadata={"found_terms": found_terms, "prohibited_terms": self.prohibited_terms},
            )

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during content validation: {str(e)}",
                metadata={"error": str(e), "prohibited_terms": self.prohibited_terms},
            )
