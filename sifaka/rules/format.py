import json
from typing import Literal, Optional, Dict, Any
from sifaka.rules.base import Rule, RuleResult


class FormatRule(Rule):
    """
    Rule that checks if the output adheres to a specific format.

    This rule is part of the Sifaka validation framework and implements format-specific
    validation logic. It supports three format types:
    - markdown: Checks for basic markdown syntax elements
    - json: Validates JSON parsing
    - plain_text: Ensures non-empty content

    Architecture Notes:
    - Inherits from the base Rule class to implement the validation contract
    - Uses private helper methods (_validate_markdown, _validate_json) to encapsulate
      format-specific validation logic
    - Returns RuleResult objects containing validation status, messages, and metadata
    - Follows the single responsibility principle by focusing only on format validation

    Data Flow:
    1. User creates FormatRule with desired format type
    2. validate() method receives output text
    3. Appropriate format-specific validator is called
    4. Result is wrapped in RuleResult with relevant metadata
    5. RuleResult is returned to the caller

    Usage Example:
        rule = FormatRule(
            name="format_rule",
            description="Validates markdown formatting",
            config={"required_format": "markdown"}
        )
    """

    required_format: Literal["markdown", "plain_text", "json"] = "plain_text"

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the rule with a required format.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration dictionary containing:
                   - required_format: The format that the output must adhere to.
                                    Options: "markdown", "plain_text", "json"
            **kwargs: Additional arguments

        Raises:
            ValueError: If required_format is not one of the allowed values
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        # Extract format from config
        config = config or {}
        required_format = config.get("required_format", "plain_text")

        if required_format not in ["markdown", "plain_text", "json"]:
            raise ValueError("required_format must be one of: markdown, plain_text, json")

        # Set the value using object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "required_format", required_format)

    def validate(self, output: str) -> RuleResult:
        """
        Validate that the output matches the required format.

        This method implements the core validation logic by:
        1. Selecting the appropriate format validator
        2. Executing the validation
        3. Constructing a detailed result message
        4. Packaging the result with relevant metadata

        Args:
            output: The text to validate

        Returns:
            RuleResult: Contains:
                       - passed: Boolean indicating validation success
                       - message: Human-readable validation result
                       - metadata: Additional validation details
        """
        try:
            if self.required_format == "markdown":
                passed = self._validate_markdown(output)
            elif self.required_format == "json":
                passed = self._validate_json(output)
            else:  # plain_text
                passed = bool(output.strip())  # Any non-empty string is valid plain text

            message = f"Output {'is' if passed else 'is not'} valid {self.required_format}"

            return RuleResult(
                passed=passed,
                message=message,
                metadata={"format": self.required_format, "output_length": len(output)},
            )

        except Exception as e:
            return RuleResult(
                passed=False,
                message=f"Error during format validation: {str(e)}",
                metadata={"error": str(e), "format": self.required_format},
            )

    def _validate_markdown(self, output: str) -> bool:
        """
        Check if the output contains basic markdown syntax.

        This is a basic validation that checks for common markdown elements:
        - Headings (#)
        - Emphasis (*, _)
        - Code blocks (`)
        - Blockquotes (>)
        - Lists (-, 1.)
        - Links ([], ())

        Note: This is a simple validation and does not guarantee valid markdown.
        For production use, consider using a markdown parser library.

        Args:
            output: The text to check for markdown syntax

        Returns:
            bool: True if basic markdown elements are found, False otherwise
        """
        markdown_elements = ["#", "*", "_", "`", ">", "-", "1.", "[", "]", "(", ")"]
        return any(element in output for element in markdown_elements)

    def _validate_json(self, output: str) -> bool:
        """
        Check if the output is valid JSON.

        This method attempts to parse the output as JSON using the standard
        json module. It returns True if parsing succeeds, False otherwise.

        Note: This only validates JSON syntax, not JSON schema or content.

        Args:
            output: The text to validate as JSON

        Returns:
            bool: True if valid JSON, False otherwise
        """
        try:
            json.loads(output)
            return True
        except json.JSONDecodeError:
            return False
