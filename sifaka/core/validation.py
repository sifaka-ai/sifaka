"""Enhanced runtime validation utilities using Pydantic.

This module extends the basic validation with more comprehensive
type checking, better error messages, and validation for additional
components throughout the Sifaka system.
"""

from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from .constants import DEFAULT_MAX_ITERATIONS
from .models import SifakaResult
from .types import CriticType, StorageType, ValidatorType


class ConfigValidationParams(BaseModel):
    """Validated parameters for configuration objects."""

    model: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        description="LLM model identifier",
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Generation temperature (0.0 to 2.0)",
    )

    max_iterations: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Maximum improvement iterations",
    )

    timeout_seconds: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=3600.0,
        description="Timeout in seconds (0.1 to 3600)",
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name format."""
        if not v.strip():
            raise ValueError("Model name cannot be empty")

        # Check for common model patterns
        valid_patterns = ["gpt-", "claude-", "gemini-", "llama-", "mistral-", "phi-"]

        if not any(v.lower().startswith(pattern) for pattern in valid_patterns):
            # Warning for unknown models, but don't fail
            pass

        return v.strip()

    model_config = {
        "extra": "allow",
        "validate_assignment": True,
    }


class ToolValidationParams(BaseModel):
    """Validated parameters for tool operations."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Search query for tools",
    )

    max_results: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum results to return",
    )

    timeout: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=300.0,
        description="Tool operation timeout in seconds",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate search query."""
        if not v.strip():
            raise ValueError("Query cannot be empty or just whitespace")

        # Check for potentially problematic characters
        if any(char in v for char in ["<", ">", "&", "'"]):
            # Clean but don't fail
            v = v.replace("<", "").replace(">", "").replace("&", "and")

        return v.strip()

    model_config = {
        "extra": "allow",
        "validate_assignment": True,
    }


class StorageValidationParams(BaseModel):
    """Validated parameters for storage operations."""

    storage_type: StorageType = Field(
        default=StorageType.MEMORY,
        description="Storage backend type",
    )

    storage_path: Optional[str] = Field(
        default=None,
        description="Path for file-based storage",
    )

    max_files: Optional[int] = Field(
        default=None,
        ge=1,
        le=100000,
        description="Maximum number of files to store",
    )

    ttl_seconds: Optional[int] = Field(
        default=None,
        ge=60,
        le=31536000,  # 1 year
        description="Time-to-live in seconds",
    )

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate storage path if provided."""
        if v is None:
            return None

        if not v.strip():
            raise ValueError("Storage path cannot be empty if provided")

        # Basic path validation
        invalid_chars = ["<", ">", "|", "?", "*"]
        if any(char in v for char in invalid_chars):
            raise ValueError(
                f"Storage path contains invalid characters: {invalid_chars}"
            )

        return v.strip()

    model_config = {
        "extra": "allow",
        "validate_assignment": True,
    }


class CriticValidationParams(BaseModel):
    """Validated parameters for critic operations."""

    critic_type: CriticType = Field(
        ...,
        description="Type of critic to use",
    )

    enable_tools: bool = Field(
        default=False,
        description="Enable tool usage for this critic",
    )

    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for critic decisions",
    )

    max_suggestions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of suggestions to generate",
    )

    custom_prompt: Optional[str] = Field(
        default=None,
        max_length=50000,
        description="Custom prompt for the critic",
    )

    @field_validator("custom_prompt")
    @classmethod
    def validate_custom_prompt(cls, v: Optional[str]) -> Optional[str]:
        """Validate custom prompt if provided."""
        if v is None:
            return None

        if not v.strip():
            return None

        # Check for reasonable length
        if len(v.strip()) < 10:
            raise ValueError("Custom prompt must be at least 10 characters")

        return v.strip()

    model_config = {
        "extra": "allow",
        "validate_assignment": True,
    }


class ValidatorValidationParams(BaseModel):
    """Validated parameters for validator operations."""

    validator_type: ValidatorType = Field(
        ...,
        description="Type of validator to use",
    )

    min_length: Optional[int] = Field(
        default=None,
        ge=0,
        le=1000000,
        description="Minimum text length",
    )

    max_length: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000000,
        description="Maximum text length",
    )

    required_terms: Optional[List[str]] = Field(
        default=None,
        description="Terms that must be present in the text",
    )

    forbidden_terms: Optional[List[str]] = Field(
        default=None,
        description="Terms that must not be present in the text",
    )

    @field_validator("required_terms")
    @classmethod
    def validate_required_terms(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate required terms list."""
        if v is None:
            return None

        if not v:
            return None

        # Filter out empty or whitespace-only terms
        valid_terms = [term.strip() for term in v if term.strip()]

        if not valid_terms:
            return None

        # Check for reasonable term length
        for term in valid_terms:
            if len(term) > 1000:
                raise ValueError(f"Term too long: {term[:50]}...")

        return valid_terms

    @field_validator("forbidden_terms")
    @classmethod
    def validate_forbidden_terms(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate forbidden terms list."""
        if v is None:
            return None

        if not v:
            return None

        # Filter out empty or whitespace-only terms
        valid_terms = [term.strip() for term in v if term.strip()]

        if not valid_terms:
            return None

        # Check for reasonable term length
        for term in valid_terms:
            if len(term) > 1000:
                raise ValueError(f"Term too long: {term[:50]}...")

        return valid_terms

    @model_validator(mode="after")
    def validate_length_constraints(self) -> "ValidatorValidationParams":
        """Validate length constraints are reasonable."""
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError(
                f"min_length ({self.min_length}) cannot be greater than "
                f"max_length ({self.max_length})"
            )

        return self

    model_config = {
        "extra": "allow",
        "validate_assignment": True,
    }


class ResultValidationParams(BaseModel):
    """Validated parameters for result storage and retrieval."""

    result_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique identifier for the result",
    )

    @field_validator("result_id")
    @classmethod
    def validate_result_id(cls, v: str) -> str:
        """Validate result ID format."""
        if not v.strip():
            raise ValueError("Result ID cannot be empty")

        # Check for basic UUID-like format or alphanumeric
        import re

        if not re.match(r"^[a-zA-Z0-9\-_]+$", v):
            raise ValueError(
                "Result ID must contain only alphanumeric characters, hyphens, and underscores"
            )

        return v.strip()

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
    }


# Enhanced validation functions with better error messages


def validate_config_params(**kwargs: Any) -> ConfigValidationParams:
    """Validate configuration parameters with enhanced error messages.

    Args:
        **kwargs: Configuration parameters to validate

    Returns:
        Validated configuration parameters

    Raises:
        ValueError: If any parameter is invalid, with detailed error message
    """
    try:
        return ConfigValidationParams(**kwargs)
    except Exception as e:
        raise ValueError(
            f"Invalid configuration parameters: {str(e)}\n"
            f"Please check your model name, temperature (0.0-2.0), "
            f"max_iterations (1-50), and timeout_seconds (0.1-3600)"
        ) from e


def validate_tool_params(query: str, **kwargs: Any) -> ToolValidationParams:
    """Validate tool operation parameters.

    Args:
        query: Search query for the tool
        **kwargs: Additional tool parameters

    Returns:
        Validated tool parameters

    Raises:
        ValueError: If any parameter is invalid, with detailed error message
    """
    try:
        return ToolValidationParams(query=query, **kwargs)
    except Exception as e:
        raise ValueError(
            f"Invalid tool parameters: {str(e)}\n"
            f"Please check your query (1-10000 characters), "
            f"max_results (1-100), and timeout (0.1-300 seconds)"
        ) from e


def validate_storage_params(**kwargs: Any) -> StorageValidationParams:
    """Validate storage parameters.

    Args:
        **kwargs: Storage parameters to validate

    Returns:
        Validated storage parameters

    Raises:
        ValueError: If any parameter is invalid, with detailed error message
    """
    try:
        return StorageValidationParams(**kwargs)
    except Exception as e:
        raise ValueError(
            f"Invalid storage parameters: {str(e)}\n"
            f"Please check your storage_type, storage_path, "
            f"max_files (1-100000), and ttl_seconds (60-31536000)"
        ) from e


def validate_critic_params(
    critic_type: CriticType, **kwargs: Any
) -> CriticValidationParams:
    """Validate critic parameters.

    Args:
        critic_type: Type of critic to validate
        **kwargs: Additional critic parameters

    Returns:
        Validated critic parameters

    Raises:
        ValueError: If any parameter is invalid, with detailed error message
    """
    try:
        return CriticValidationParams(critic_type=critic_type, **kwargs)
    except Exception as e:
        raise ValueError(
            f"Invalid critic parameters: {str(e)}\n"
            f"Please check your critic_type, confidence_threshold (0.0-1.0), "
            f"max_suggestions (1-20), and custom_prompt (10-50000 characters)"
        ) from e


def validate_validator_params(
    validator_type: ValidatorType, **kwargs: Any
) -> ValidatorValidationParams:
    """Validate validator parameters.

    Args:
        validator_type: Type of validator to validate
        **kwargs: Additional validator parameters

    Returns:
        Validated validator parameters

    Raises:
        ValueError: If any parameter is invalid, with detailed error message
    """
    try:
        return ValidatorValidationParams(validator_type=validator_type, **kwargs)
    except Exception as e:
        raise ValueError(
            f"Invalid validator parameters: {str(e)}\n"
            f"Please check your validator_type, length constraints, "
            f"required_terms, and forbidden_terms"
        ) from e


def validate_result_id(result_id: str) -> str:
    """Validate result ID format.

    Args:
        result_id: Result ID to validate

    Returns:
        Validated result ID

    Raises:
        ValueError: If result ID is invalid, with detailed error message
    """
    try:
        validated = ResultValidationParams(result_id=result_id)
        return validated.result_id
    except Exception as e:
        raise ValueError(
            f"Invalid result ID: {str(e)}\n"
            f"Result ID must be 1-255 characters long and contain only "
            f"alphanumeric characters, hyphens, and underscores"
        ) from e


def validate_sifaka_result(result: Any) -> SifakaResult:
    """Validate that an object is a proper SifakaResult.

    Args:
        result: Object to validate

    Returns:
        Validated SifakaResult

    Raises:
        ValueError: If object is not a valid SifakaResult
    """
    if not isinstance(result, SifakaResult):
        raise ValueError(
            f"Expected SifakaResult, got {type(result).__name__}. "
            f"Please ensure you're working with a valid Sifaka result object."
        )

    # Additional validation of result contents
    if not result.original_text:
        raise ValueError("SifakaResult must have non-empty original_text")

    if not result.final_text:
        raise ValueError("SifakaResult must have non-empty final_text")

    if result.iteration < 0:
        raise ValueError("SifakaResult iteration must be non-negative")

    if result.processing_time < 0:
        raise ValueError("SifakaResult processing_time must be non-negative")

    return result


# Type guard functions for common patterns


def is_valid_model_name(model: str) -> bool:
    """Check if a model name is valid.

    Args:
        model: Model name to check

    Returns:
        True if valid, False otherwise
    """
    try:
        validate_config_params(model=model)
        return True
    except ValueError:
        return False


def is_valid_temperature(temperature: float) -> bool:
    """Check if a temperature value is valid.

    Args:
        temperature: Temperature value to check

    Returns:
        True if valid, False otherwise
    """
    try:
        validate_config_params(temperature=temperature)
        return True
    except ValueError:
        return False


def is_valid_result_id(result_id: str) -> bool:
    """Check if a result ID is valid.

    Args:
        result_id: Result ID to check

    Returns:
        True if valid, False otherwise
    """
    try:
        validate_result_id(result_id)
        return True
    except ValueError:
        return False


# Additional validation functions for API compatibility


class ImproveParams(BaseModel):
    """Validated parameters for the improve() function."""

    text: str = Field(
        ...,
        min_length=1,
        description="Text to improve. Cannot be empty.",
    )

    critics: Optional[List[CriticType]] = Field(
        default=None,
        description="Critics to use for text improvement",
    )

    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS,
        ge=1,
        le=100,
        description="Maximum improvement iterations",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure text is not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or just whitespace")
        return v

    @field_validator("critics", mode="before")
    @classmethod
    def validate_critics(cls, v: Any) -> Optional[List[CriticType]]:
        """Normalize critics to a list and validate types."""
        if v is None:
            return None

        if isinstance(v, CriticType):
            return [v]

        if isinstance(v, str):
            # Single string critic
            try:
                return [CriticType(v)]
            except ValueError:
                valid_values = ", ".join(CriticType.values())
                raise ValueError(
                    f"Invalid critic type: '{v}'. Must be one of: {valid_values}"
                )

        if not isinstance(v, list):
            raise ValueError(
                f"Critics must be CriticType, string, or list, got {type(v).__name__}"
            )

        # Validate each critic
        result = []
        for critic in v:
            if isinstance(critic, CriticType):
                result.append(critic)
            elif isinstance(critic, str):
                # Try to convert string to CriticType
                try:
                    result.append(CriticType(critic))
                except ValueError:
                    valid_values = ", ".join(CriticType.values())
                    raise ValueError(
                        f"Invalid critic type: '{critic}'. Must be one of: {valid_values}"
                    )
            else:
                raise ValueError(
                    f"Invalid critic type: {critic}. Must be CriticType enum value or string."
                )

        return result

    model_config = {
        "extra": "forbid",  # Reject unknown parameters
        "validate_assignment": True,
    }


def validate_improve_params(
    text: str,
    critics: Optional[Union[CriticType, List[CriticType]]] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> ImproveParams:
    """Validate parameters for the improve() function.

    Args:
        text: Text to improve
        critics: Critics to use
        max_iterations: Maximum iterations

    Returns:
        Validated parameters

    Raises:
        ValueError: If any parameter is invalid
    """
    return ImproveParams(
        text=text,
        critics=critics,
        max_iterations=max_iterations,
    )
