"""
Length validation rules for Sifaka.
"""

from typing import Optional, Dict, Any
from sifaka.rules.base import Rule, RuleResult
from pydantic import Field


class LengthRule(Rule):
    """
    Rule that checks if the output length falls within specified bounds.

    This rule supports both character and word count validation.

    Attributes:
        min_length: Minimum length
        max_length: Maximum length (optional)
        exact_length: Exact length required (optional)
        unit: Unit of measurement ('characters' or 'words')
    """

    min_length: int = Field(default=50, description="Minimum length")
    max_length: Optional[int] = Field(default=5000, description="Maximum length")
    exact_length: Optional[int] = Field(default=None, description="Exact length required")
    unit: str = Field(default="characters", description="Unit of measurement (characters or words)")

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the rule with length constraints.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration dictionary containing:
                   - min_length: Minimum allowed length (inclusive)
                   - max_length: Maximum allowed length (inclusive)
                   - exact_length: Exact required length
                   - unit: Unit of measurement ('characters' or 'words')
            **kwargs: Additional arguments
        """
        super().__init__(name=name, description=description, config=config or {}, **kwargs)

        config = config or {}
        self.min_length = config.get("min_length", self.min_length)
        self.max_length = config.get("max_length", self.max_length)
        self.exact_length = config.get("exact_length", self.exact_length)
        self.unit = config.get("unit", self.unit)

        if self.unit not in ["characters", "words"]:
            raise ValueError("Unit must be either 'characters' or 'words'")

        if self.exact_length is not None:
            if self.exact_length < 0:
                raise ValueError("exact_length must be non-negative")
            if self.min_length != 50 or self.max_length != 5000:
                raise ValueError("exact_length cannot be used with min_length or max_length")
        else:
            if self.min_length < 0:
                raise ValueError("min_length must be non-negative")
            if self.max_length is not None and self.max_length < 0:
                raise ValueError("max_length must be non-negative")
            if self.max_length is not None and self.min_length > self.max_length:
                raise ValueError("min_length cannot be greater than max_length")

    def _get_length(self, text: str) -> int:
        """
        Get the length of the text in the specified unit.

        Args:
            text: The text to measure

        Returns:
            Length in the specified unit
        """
        if self.unit == "words":
            return len(text.split())
        return len(text)

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Validate that the output length is within acceptable bounds.

        Args:
            output: The text to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results
        """
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        length = self._get_length(output)
        metadata = {
            "length": length,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "exact_length": self.exact_length,
            "unit": self.unit,
        }

        # Handle empty or whitespace-only text
        if not output.strip():
            return RuleResult(
                passed=False,
                message=f"Empty or whitespace-only text (0 {self.unit})",
                metadata=metadata,
            )

        # Check exact length if specified
        if self.exact_length is not None:
            if length != self.exact_length:
                return RuleResult(
                    passed=False,
                    message=f"Output {self.unit} count {length} does not match required count of {self.exact_length}",
                    metadata=metadata,
                )
            return RuleResult(
                passed=True,
                message=f"Output {self.unit} count matches required count of {self.exact_length}",
                metadata=metadata,
            )

        # Check length bounds
        issues = []
        if length < self.min_length:
            issues.append(f"below minimum of {self.min_length}")
        if self.max_length is not None and length > self.max_length:
            issues.append(f"exceeds maximum of {self.max_length}")

        if issues:
            return RuleResult(
                passed=False,
                message=f"Output {self.unit} count {length} {' and '.join(issues)}",
                metadata=metadata,
            )

        return RuleResult(
            passed=True,
            message=f"Output {self.unit} count {length} meets requirements",
            metadata=metadata,
        )
