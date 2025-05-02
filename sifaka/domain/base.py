"""Base domain configuration and factory methods."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class DomainConfig(BaseModel):
    """Base domain configuration."""

    name: str = Field(..., description="Name of the domain configuration")
    description: str = Field(..., description="Description of the domain configuration")
    params: Dict[str, Any] = Field(default_factory=dict, description="Domain parameters")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate the name field."""
        if not v:
            raise ValueError("Name cannot be empty")
        return v

    @classmethod
    def create_with_config(cls, config: Dict[str, Any]) -> "DomainConfig":
        """Create a domain configuration from a dictionary."""
        return cls(**config)

    @classmethod
    def create(cls, config: Dict[str, Any]) -> "DomainConfig":
        """Create a domain configuration from a dictionary."""
        return cls.create_with_config(config)

    @classmethod
    def create_with_params(
        cls,
        name: str,
        description: str,
        params: Optional[Dict[str, Any]] = None
    ) -> "DomainConfig":
        """Create a domain configuration from individual parameters."""
        return cls(
            name=name,
            description=description,
            params=params or {}
        )