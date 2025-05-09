"""
Validation module for Sifaka.

This module provides the Validator class which is responsible for
validating text against rules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from sifaka.rules.base import Rule
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class ValidatorConfig(BaseModel):
    """
    Configuration for validators.

    This class represents the configuration for a validator, including
    rules, validation mode, and other settings.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
        validate_assignment=True,
    )

    rules: List[Rule] = Field(
        default_factory=list,
        description="List of rules to validate against",
    )
    fail_fast: bool = Field(
        default=False,
        description="Whether to stop validation after the first failure",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the validator",
    )


@dataclass
class ValidationResult(Generic[OutputType]):
    """
    Result of a validation operation.

    This class represents the result of a validation operation, including
    the validation status, rule results, and additional metadata.
    """

    output: OutputType
    passed: bool
    rule_results: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Validator(Generic[InputType, OutputType]):
    """
    Handles validation of text against rules.

    This class is responsible for validating text against rules.
    It provides a consistent interface for validation across different
    rule types.
    """

    def __init__(self, config: Optional[ValidatorConfig] = None):
        """
        Initialize a Validator instance.

        Args:
            config: Optional configuration for the validator
        """
        self._config = config or ValidatorConfig()

    @property
    def config(self) -> ValidatorConfig:
        """
        Get the validator configuration.

        Returns:
            The validator configuration
        """
        return self._config

    def validate(self, input_value: InputType, output_value: OutputType) -> ValidationResult[OutputType]:
        """
        Validate output against rules.

        Args:
            input_value: The input value that produced the output
            output_value: The output value to validate

        Returns:
            The validation result

        Raises:
            TypeError: If input_value or output_value is of the wrong type
            ValueError: If validation fails
        """
        # This is a mock implementation that always passes
        return ValidationResult[OutputType](
            output=output_value,
            passed=True,
            rule_results=[],
            metadata={"mock": True},
        )
