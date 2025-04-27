from sifaka.rules.base import Rule, RuleResult
from typing import Any, Optional


class TemplateRule(Rule):
    """
    Template for creating new rules in Sifaka.

    This class demonstrates the structure and best practices for creating new rules.
    Replace the docstring and implementation with your rule's specific functionality.
    """

    def __init__(self, param1: Any, param2: Optional[Any] = None):
        """
        Initialize the rule with its parameters.

        Args:
            param1: Description of the first parameter
            param2: Description of the second parameter (optional)
        """
        super().__init__(
            name="template_rule",  # Replace with your rule's name
            description="Description of what this rule checks for",  # Replace with your rule's description
        )
        self.param1 = param1
        self.param2 = param2

    def validate(self, output: str) -> RuleResult:
        """
        Validate the output against the rule's criteria.

        Args:
            output: The text to validate

        Returns:
            RuleResult: The result of the validation

        Raises:
            ValueError: If output is None
        """
        if output is None:
            raise ValueError("Output cannot be None")

        try:
            # Implement your validation logic here
            # Example: Check if output meets some criteria
            if not self._check_criteria(output):
                return RuleResult(
                    passed=False,
                    message="Description of why the validation failed",
                    metadata={
                        "param1": self.param1,
                        "param2": self.param2,
                        # Add any additional metadata that might be useful
                    },
                )

            return RuleResult(passed=True)

        except Exception as e:
            # Handle any exceptions that might occur during validation
            return RuleResult(
                passed=False,
                message=f"Error during validation: {str(e)}",
                metadata={"error": str(e)},
            )

    def _check_criteria(self, output: str) -> bool:
        """
        Helper method to check specific criteria.
        Replace with your actual validation logic.

        Args:
            output: The text to check

        Returns:
            bool: True if criteria are met, False otherwise
        """
        # Implement your specific validation logic here
        return True  # Replace with actual validation
