"""
Base Configuration Module

This module provides the base configuration class for all Sifaka components.

## Overview
The base configuration module defines the foundation for all configuration classes
in the Sifaka framework. It provides a consistent approach to configuration with
standardized parameter handling, validation, and serialization.

## Components
- **BaseConfig**: Base configuration class for all components
- Type variables for generic configuration handling

## Usage Examples
```python
from sifaka.utils.config.base import BaseConfig

# Create a basic configuration
config = BaseConfig(
    name="my_component",
    description="A custom component",
    params={"threshold": 0.7}
)

# Access configuration values
print(f"Name: {config.name}")
print(f"Custom threshold: {config.params.get('threshold')}")

# Create a new configuration with updated parameters
updated_config = config.with_params(max_length=100, min_length=10)

# Create a new configuration with updated options
updated_config = config.with_options(name="new_name")
```

## Error Handling
The configuration utilities use Pydantic for validation, which ensures that
configuration values are valid and properly typed. If invalid configuration
is provided, Pydantic will raise validation errors with detailed information
about the validation failure.
"""

from typing import Any, Dict, TypeVar
from pydantic import BaseModel, Field, ConfigDict

# Type variables for generic configuration handling
T = TypeVar("T", bound=BaseModel)


class BaseConfig(BaseModel):
    """
    Base configuration for all Sifaka components.

    This class provides a consistent foundation for all configuration classes
    in the Sifaka framework. It defines common fields and methods that are
    shared across all component types.

    ## Architecture
    BaseConfig uses Pydantic for validation and serialization, with:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during component initialization and
    remain immutable throughout the component's lifecycle. Components can access
    configuration values through their config property.

    ## Examples
    ```python
    # Create a basic configuration
    config = BaseConfig(
        name="my_component",
        description="A custom component",
        params={"threshold": 0.7}
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Custom threshold: {config.params.get('threshold')}")

    # Create a new configuration with updated parameters
    updated_config = config.with_params(max_length=100, min_length=10)

    # Create a new configuration with updated options
    updated_config = config.with_options(name="new_name")
    ```

    Attributes:
        name: Component name
        description: Component description
        params: Dictionary of additional parameters
    """

    name: str = Field(default="", description="Component name")
    description: str = Field(default="", description="Component description")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

    model_config = ConfigDict(frozen=True)

    def with_params(self, **kwargs: Any) -> "BaseConfig":
        """
        Create a new configuration with updated parameters.

        This method creates a new configuration object with the same options as the
        current configuration, but with updated parameters. The original configuration
        remains unchanged due to the immutable nature of configuration objects.

        Args:
            **kwargs: Parameters to update in the params dictionary

        Returns:
            New configuration with updated parameters

        Example:
            ```python
            # Create a configuration with parameters
            config = BaseConfig(
                name="my_component",
                params={"threshold": 0.7}
            )

            # Create a new configuration with updated parameters
            updated_config = config.with_params(
                threshold=0.8,
                max_length=100
            )

            # Original config is unchanged
            assert config.params["threshold"] == 0.7
            assert "max_length" not in config.params

            # New config has updated parameters
            assert updated_config.params["threshold"] == 0.8
            assert updated_config.params["max_length"] == 100
            ```
        """
        return self.model_copy(update={"params": {**self.params, **kwargs}})

    def with_options(self, **kwargs: Any) -> "BaseConfig":
        """
        Create a new configuration with updated options.

        This method creates a new configuration object with updated options.
        Unlike with_params, which updates the params dictionary, this method
        updates the configuration fields directly. The original configuration
        remains unchanged due to the immutable nature of configuration objects.

        Args:
            **kwargs: Configuration options to update

        Returns:
            New configuration with updated options

        Example:
            ```python
            # Create a configuration
            config = BaseConfig(
                name="my_component",
                description="Original description",
                params={"threshold": 0.7}
            )

            # Create a new configuration with updated options
            updated_config = config.with_options(
                name="new_name",
                description="Updated description"
            )

            # Original config is unchanged
            assert config.name == "my_component"
            assert config.description == "Original description"

            # New config has updated options
            assert updated_config.name == "new_name"
            assert updated_config.description == "Updated description"

            # Params are preserved
            assert updated_config.params == config.params
            ```
        """
        return self.model_copy(update=kwargs)
