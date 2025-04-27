from typing import List
from sifaka.rules.base import Rule, RuleResult


class ProhibitedContentRule(Rule):
    """
    Rule that checks if the output contains any prohibited terms.

    This rule is part of the Sifaka validation framework and implements content
    filtering by checking for prohibited terms in text output. It performs
    case-insensitive matching to ensure comprehensive content validation.

    Architecture Notes:
    - Inherits from the base Rule class to implement the validation contract
    - Uses case-insensitive matching for comprehensive term detection
    - Returns RuleResult objects containing validation status, messages, and metadata
    - Follows the single responsibility principle by focusing only on prohibited content
    - Designed to be efficient with large lists of prohibited terms

    Data Flow:
    1. User creates ProhibitedContentRule with list of prohibited terms
    2. validate() method receives output text
    3. Text is checked for any prohibited terms (case-insensitive)
    4. Result is wrapped in RuleResult with relevant metadata
    5. RuleResult is returned to the caller

    Usage Example:
        rule = ProhibitedContentRule(["bad", "inappropriate", "forbidden"])
        result = rule.validate("This is a bad example")
        # result.passed will be False
        # result.message will indicate which terms were found
    """

    def __init__(self, prohibited_terms: List[str]):
        """
        Initialize the rule with a list of prohibited terms.

        Args:
            prohibited_terms: List of terms that should not appear in the output.
                            Terms are matched case-insensitively.
                            Example: ["bad", "inappropriate", "forbidden"]

        Raises:
            ValueError: If prohibited_terms list is empty
        """
        if not prohibited_terms:
            raise ValueError("prohibited_terms list cannot be empty")

        super().__init__(
            name="prohibited_content_rule",
            description=f"Checks if output contains any of these prohibited terms: {', '.join(prohibited_terms)}",
        )
        self.prohibited_terms = prohibited_terms

    def validate(self, output: str) -> RuleResult:
        """
        Validate that the output does not contain any prohibited terms.

        This method implements the core validation logic by:
        1. Converting both output and terms to lowercase for case-insensitive matching
        2. Checking for any prohibited terms in the output
        3. Constructing a detailed result message listing found terms
        4. Packaging the result with relevant metadata

        Args:
            output: The text to validate

        Returns:
            RuleResult: Contains:
                       - passed: Boolean indicating if no prohibited terms were found
                       - message: Human-readable validation result listing found terms
                       - metadata: Additional validation details including found terms

        Note: Matching is case-insensitive. For example, "BAD" will match "bad".
        """
        try:
            found_terms = [term for term in self.prohibited_terms if term.lower() in output.lower()]
            passed = not found_terms

            message = (
                f"Output {'contains' if found_terms else 'does not contain'} "
                f"prohibited terms: {', '.join(found_terms) if found_terms else 'none'}"
            )

            return RuleResult(
                passed=passed,
                message=message,
                metadata={"found_terms": found_terms, "prohibited_terms": self.prohibited_terms},
            )

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during prohibited content validation: {str(e)}",
                metadata={"error": str(e), "prohibited_terms": self.prohibited_terms},
            )
