"""Validation-specific configuration."""

from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from ..type_defs import ValidatorSettings
from ..types import ValidatorType
from .base import BaseConfig


class ValidationConfig(BaseConfig):
    """Configuration for validators and validation behavior.

    Controls which validators are used and their specific parameters.
    Validators ensure text meets specific requirements before and after
    improvement.

    Example:
        >>> validation_config = ValidationConfig(
        ...     validators=["length", "format"],
        ...     validator_settings={
        ...         "length": {"min_length": 100, "max_length": 500},
        ...         "format": {"min_paragraphs": 2}
        ...     }
        ... )
    """

    # Validator selection
    validators: Optional[List[ValidatorType]] = Field(
        default=None, description="List of validators to apply (None = no validation)"
    )

    @field_validator("validators", mode="before")
    @classmethod
    def validate_validators(cls, v: Any) -> Optional[List[ValidatorType]]:
        """Validate validator types - ONLY ValidatorType enums allowed."""
        if v is None:
            return None

        if isinstance(v, ValidatorType):
            v = [v]

        if not isinstance(v, list):
            raise ValueError(
                f"validators must be a list of ValidatorType enums, got {type(v).__name__}"
            )

        result: List[ValidatorType] = []
        for validator in v:
            if isinstance(validator, ValidatorType):
                result.append(validator)
            else:
                available = ", ".join(
                    f"ValidatorType.{vt.name}" for vt in ValidatorType
                )
                raise ValueError(
                    f"Invalid validator type: {validator} (type: {type(validator).__name__}). "
                    f"Must use ValidatorType enum values: {available}"
                )

        return result

    # Validation behavior
    stop_on_validation_failure: bool = Field(
        default=False, description="Stop improvement process if validation fails"
    )

    validate_improvements: bool = Field(
        default=True, description="Validate each improvement iteration"
    )

    # Per-validator settings
    validator_settings: Dict[str, ValidatorSettings] = Field(
        default_factory=dict, description="Configuration for specific validators"
    )

    # Common validator parameters
    length_min: Optional[int] = Field(
        default=None, gt=0, description="Minimum text length in characters"
    )

    length_max: Optional[int] = Field(
        default=None, gt=0, description="Maximum text length in characters"
    )

    format_min_paragraphs: Optional[int] = Field(
        default=None, gt=0, description="Minimum number of paragraphs"
    )

    format_max_paragraphs: Optional[int] = Field(
        default=None, gt=0, description="Maximum number of paragraphs"
    )

    content_required_terms: Optional[List[str]] = Field(
        default=None, description="Terms that must appear in the text"
    )

    content_forbidden_terms: Optional[List[str]] = Field(
        default=None, description="Terms that must not appear in the text"
    )

    def get_validator_settings(self, validator_name: str) -> ValidatorSettings:
        """Get settings for a specific validator.

        Args:
            validator_name: Name of the validator

        Returns:
            Dictionary of settings for the validator
        """
        # Start with explicit validator settings
        settings = self.validator_settings.get(validator_name, {}).copy()

        # Add common parameters based on validator type
        if validator_name == "length":
            if self.length_min is not None and "min_length" not in settings:
                settings["min_length"] = self.length_min
            if self.length_max is not None and "max_length" not in settings:
                settings["max_length"] = self.length_max

        elif validator_name == "format":
            if (
                self.format_min_paragraphs is not None
                and "min_paragraphs" not in settings
            ):
                settings["min_paragraphs"] = self.format_min_paragraphs
            if (
                self.format_max_paragraphs is not None
                and "max_paragraphs" not in settings
            ):
                settings["max_paragraphs"] = self.format_max_paragraphs

        elif validator_name == "content":
            if (
                self.content_required_terms is not None
                and "required_terms" not in settings
            ):
                settings["required_terms"] = self.content_required_terms
            if (
                self.content_forbidden_terms is not None
                and "forbidden_terms" not in settings
            ):
                settings["forbidden_terms"] = self.content_forbidden_terms

        return settings

    def has_validators(self) -> bool:
        """Check if any validators are configured."""
        return self.validators is not None and len(self.validators) > 0
