"""
Guardrails validator adapter for using guardrails as validators.

This module provides a validator that adapts Guardrail components to be used
as Validators in the Sifaka system.
"""

from typing import Any, Dict, List, Optional, Union

from ..types import ValidationResult
from ..guardrails import Guardrail, GuardrailResult


class GuardrailsValidator:
    """
    Validator that adapts a Guardrail to be used as a Validator.

    This validator wraps a Guardrail component and provides a Validator interface
    that can be used in a Chain.
    """

    def __init__(
        self,
        guardrail: Guardrail,
        fail_on_any_issue: bool = True,
    ):
        """
        Initialize the GuardrailsValidator.

        Args:
            guardrail: The guardrail to use for validation
            fail_on_any_issue: If True, validation fails if any issues are found
        """
        self.guardrail = guardrail
        self.fail_on_any_issue = fail_on_any_issue

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text using the guardrail.

        Args:
            text: The text to validate

        Returns:
            ValidationResult indicating whether the text passes validation
        """
        # Use the guardrail's run method if available, otherwise use process and validate
        if hasattr(self.guardrail, "run"):
            guardrail_result = self.guardrail.run(text)
        else:
            processed_text = self.guardrail.process(text)
            passed = self.guardrail.validate(text)
            guardrail_result = GuardrailResult(
                processed_text=processed_text,
                passed=passed,
                message="Guardrail validation result",
            )

        # Map guardrail result to validation result
        return ValidationResult(
            passed=guardrail_result.passed,
            message=guardrail_result.message,
            score=1.0 if guardrail_result.passed else 0.0,
            issues=guardrail_result.issues,
            suggestions=[
                f"Consider: {processed_text}"
                for processed_text in [guardrail_result.processed_text]
                if processed_text != text
            ],
            metadata=guardrail_result.metadata,
        )
