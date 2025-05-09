"""
Result formatter module for Sifaka.

This module provides the ResultFormatter class which is responsible for
formatting and processing results.
"""

from typing import Any, Dict, Generic, Optional, TypeVar

from ..interfaces.formatter import ResultFormatterProtocol
from ...validation import ValidationResult
from ..result import ChainResult
from ...utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class ResultFormatter(
    ResultFormatterProtocol[ValidationResult[OutputType], ChainResult[OutputType]],
    Generic[OutputType],
):
    """
    Formats and processes results.

    This class is responsible for formatting and processing results from
    chain execution. It implements the ResultFormatterProtocol interface.
    """

    def format_result(
        self,
        output: OutputType,
        validation_result: ValidationResult[OutputType],
        critique_details: Optional[Dict[str, Any]] = None,
    ) -> ChainResult[OutputType]:
        """
        Format a result.

        Args:
            output: The output
            validation_result: The validation result
            critique_details: Optional critique details

        Returns:
            The formatted result
        """
        return ChainResult(
            output=output,
            rule_results=validation_result.rule_results,
            critique_details=critique_details,
        )

    def format_feedback_from_validation(
        self, validation_result: ValidationResult[OutputType]
    ) -> str:
        """
        Format feedback from a validation result.

        Args:
            validation_result: The validation result

        Returns:
            The formatted feedback
        """
        feedback = "The following issues were found:\n"
        for result in validation_result.rule_results:
            if not result.passed:
                feedback += f"- {result.message}\n"
        return feedback

    def format_feedback_from_critique(self, critique_details: Dict[str, Any]) -> str:
        """
        Format feedback from critique details.

        Args:
            critique_details: The critique details

        Returns:
            The formatted feedback
        """
        if "feedback" in critique_details:
            return critique_details["feedback"]

        feedback = "The following issues were found:\n"

        if "issues" in critique_details and critique_details["issues"]:
            for issue in critique_details["issues"]:
                feedback += f"- {issue}\n"

        if "suggestions" in critique_details and critique_details["suggestions"]:
            feedback += "\nSuggestions for improvement:\n"
            for suggestion in critique_details["suggestions"]:
                feedback += f"- {suggestion}\n"

        return feedback

    def format(self, result: ValidationResult[OutputType]) -> ChainResult[OutputType]:
        """
        Format a result.

        This method implements the ResultFormatterProtocol.format method.

        Args:
            result: The result to format

        Returns:
            A formatted result

        Raises:
            ValueError: If the result is invalid
        """
        if not isinstance(result, ValidationResult):
            raise ValueError(f"Expected ValidationResult, got {type(result)}")

        return ChainResult(
            output=result.output,
            rule_results=result.rule_results,
            critique_details=None,
        )
