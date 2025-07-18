"""Base configuration class for all Sifaka configurations."""

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base configuration class with common settings.

    All configuration classes inherit from this to ensure consistent
    behavior and validation.
    """

    model_config = ConfigDict(
        extra="forbid",  # Prevent typos in config
        validate_assignment=True,  # Validate on attribute assignment
        use_enum_values=True,  # Use enum values instead of enum instances
    )
