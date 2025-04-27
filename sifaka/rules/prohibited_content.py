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
    - Uses case-sensitive or case-insensitive matching based on configuration
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
            config={
                "prohibited_terms": ["bad", "inappropriate", "forbidden"],
                "case_sensitive": False
            }
        )
        result = rule.validate("This is a bad example")
        # result.passed will be False
        # result.message will indicate which terms were found
    """

    prohibited_terms: List[str] = Field(
        default=[], description="List of terms that should not appear in the output"
    )
    case_sensitive: bool = Field(
        default=False, description="Whether to perform case-sensitive matching"
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
                   - case_sensitive: Whether to perform case-sensitive matching
            **kwargs: Additional arguments

        Raises:
            ValueError: If prohibited_terms list is empty
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        # Extract configuration
        config = config or {}
        prohibited_terms = config.get("prohibited_terms", [])
        case_sensitive = config.get("case_sensitive", False)

        if not prohibited_terms:
            raise ValueError("prohibited_terms list cannot be empty")

        self.prohibited_terms = prohibited_terms
        self.case_sensitive = case_sensitive

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output does not contain prohibited terms.

        Args:
            output: The text to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult indicating whether prohibited terms were found
        """
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        found_terms = []
        text = output if self.case_sensitive else output.lower()
        terms = (
            self.prohibited_terms
            if self.case_sensitive
            else [t.lower() for t in self.prohibited_terms]
        )

        for term in terms:
            if term in text:
                found_terms.append(term)

        if found_terms:
            return RuleResult(
                passed=False,
                message=f"Found prohibited terms: {', '.join(found_terms)}",
                metadata={"found_terms": found_terms},
            )

        return RuleResult(
            passed=True,
            message="No prohibited terms found",
            metadata={"found_terms": []},
        )
