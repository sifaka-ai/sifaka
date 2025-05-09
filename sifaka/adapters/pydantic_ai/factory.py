"""
PydanticAI Factory

Factory functions for creating PydanticAI adapters.

## Overview
This module provides factory functions for creating PydanticAI adapters with
various configurations. These functions simplify the creation of adapters by
providing a consistent interface and handling common configuration patterns.

## Components
1. **create_pydantic_adapter**: Creates a basic PydanticAI adapter
2. **create_pydantic_adapter_with_critic**: Creates a PydanticAI adapter with critic support

## Usage Examples
```python
from pydantic import BaseModel
from sifaka.adapters.pydantic_ai import create_pydantic_adapter
from sifaka.rules.formatting.length import create_length_rule

# Define a Pydantic model
class Response(BaseModel):
    content: str

# Create rules and adapter
rules = [create_length_rule(min_chars=10, max_chars=100)]
adapter = create_pydantic_adapter(
    rules=rules,
    output_model=Response,
    max_refine=2
)
```

## Error Handling
- ValueError: Raised when configuration is invalid
- TypeError: Raised when input types are incompatible
- ImportError: Raised when required dependencies are missing

## Configuration
- max_refine: Maximum number of refinement attempts
- prioritize_by_cost: Whether to prioritize rules by cost
- config: Optional configuration dictionary or SifakaPydanticConfig instance
"""

from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from sifaka.critics.base import BaseCritic
from sifaka.critics.prompt import create_prompt_critic
from sifaka.models.base import ModelProvider
from sifaka.rules.base import Rule

from .adapter import SifakaPydanticAdapter, SifakaPydanticConfig


def create_pydantic_adapter(
    rules: List[Rule],
    output_model: Type[BaseModel],
    max_refine: int = 2,
    prioritize_by_cost: bool = False,
    config: Optional[Union[Dict[str, Any], SifakaPydanticConfig]] = None,
    **kwargs: Any,
) -> SifakaPydanticAdapter:
    """
    Factory function to create a PydanticAI adapter.

    ## Overview
    This function creates a SifakaPydanticAdapter with the specified rules and
    configuration, providing a simple way to create adapters for common use cases.

    Args:
        rules (List[Rule]): List of Sifaka rules to validate against
        output_model (Type[BaseModel]): The Pydantic model type for the output
        max_refine (int): Maximum number of refinement attempts
        prioritize_by_cost (bool): Whether to prioritize rules by cost
        config (Optional[Union[Dict[str, Any], SifakaPydanticConfig]]): Optional configuration
        **kwargs: Additional keyword arguments for the adapter

    Returns:
        SifakaPydanticAdapter: A configured adapter instance

    Raises:
        ValueError: If configuration is invalid
        TypeError: If input types are incompatible

    ## Examples
    ```python
    from pydantic import BaseModel
    from sifaka.adapters.pydantic_ai import create_pydantic_adapter
    from sifaka.rules.formatting.length import create_length_rule

    # Define a Pydantic model
    class Response(BaseModel):
        content: str

    # Create rules and adapter
    rules = [create_length_rule(min_chars=10, max_chars=100)]
    adapter = create_pydantic_adapter(
        rules=rules,
        output_model=Response,
        max_refine=2
    )
    ```
    """
    # Process configuration
    adapter_config = None
    if config is not None:
        if isinstance(config, dict):
            # Create config from dict
            config_dict = config.copy()
            # Add explicit parameters if not in dict
            if "max_refine" not in config_dict and max_refine is not None:
                config_dict["max_refine"] = max_refine
            if "prioritize_by_cost" not in config_dict and prioritize_by_cost is not None:
                config_dict["prioritize_by_cost"] = prioritize_by_cost
            adapter_config = SifakaPydanticConfig(**config_dict)
        else:
            # Use provided config
            adapter_config = config
    else:
        # Create config from explicit parameters
        adapter_config = SifakaPydanticConfig(
            max_refine=max_refine,
            prioritize_by_cost=prioritize_by_cost,
        )

    # Create and return the adapter
    return SifakaPydanticAdapter(
        rules=rules,
        output_model=output_model,
        config=adapter_config,
        **kwargs,
    )


def create_pydantic_adapter_with_critic(
    rules: List[Rule],
    output_model: Type[BaseModel],
    critic: Optional[BaseCritic] = None,
    model_provider: Optional[ModelProvider] = None,
    system_prompt: Optional[str] = None,
    max_refine: int = 2,
    prioritize_by_cost: bool = False,
    config: Optional[Union[Dict[str, Any], SifakaPydanticConfig]] = None,
    **kwargs: Any,
) -> SifakaPydanticAdapter:
    """
    Factory function to create a PydanticAI adapter with critic support.

    ## Overview
    This function creates a SifakaPydanticAdapter with the specified rules, critic,
    and configuration, providing a simple way to create adapters for use cases
    that require refinement.

    Args:
        rules (List[Rule]): List of Sifaka rules to validate against
        output_model (Type[BaseModel]): The Pydantic model type for the output
        critic (Optional[BaseCritic]): Optional Sifaka critic for refinement
        model_provider (Optional[ModelProvider]): Optional model provider for creating a critic
        system_prompt (Optional[str]): Optional system prompt for creating a critic
        max_refine (int): Maximum number of refinement attempts
        prioritize_by_cost (bool): Whether to prioritize rules by cost
        config (Optional[Union[Dict[str, Any], SifakaPydanticConfig]]): Optional configuration
        **kwargs: Additional keyword arguments for the adapter

    Returns:
        SifakaPydanticAdapter: A configured adapter instance with critic support

    Raises:
        ValueError: If configuration is invalid
        TypeError: If input types are incompatible
        ImportError: If required dependencies are missing

    ## Examples
    ```python
    from pydantic import BaseModel
    from sifaka.adapters.pydantic_ai import create_pydantic_adapter_with_critic
    from sifaka.rules.formatting.length import create_length_rule
    from sifaka.models.factories import create_openai_provider

    # Define a Pydantic model
    class Response(BaseModel):
        content: str

    # Create model provider
    provider = create_openai_provider(model_name="gpt-4")

    # Create rules and adapter
    rules = [create_length_rule(min_chars=10, max_chars=100)]
    adapter = create_pydantic_adapter_with_critic(
        rules=rules,
        output_model=Response,
        model_provider=provider,
        system_prompt="You are an expert editor that improves text while maintaining its original meaning.",
        max_refine=2
    )
    ```
    """
    # Process configuration
    adapter_config = None
    if config is not None:
        if isinstance(config, dict):
            # Create config from dict
            config_dict = config.copy()
            # Add explicit parameters if not in dict
            if "max_refine" not in config_dict and max_refine is not None:
                config_dict["max_refine"] = max_refine
            if "prioritize_by_cost" not in config_dict and prioritize_by_cost is not None:
                config_dict["prioritize_by_cost"] = prioritize_by_cost
            adapter_config = SifakaPydanticConfig(**config_dict)
        else:
            # Use provided config
            adapter_config = config
    else:
        # Create config from explicit parameters
        adapter_config = SifakaPydanticConfig(
            max_refine=max_refine,
            prioritize_by_cost=prioritize_by_cost,
        )

    # Create critic if not provided
    if critic is None and model_provider is not None:
        # Default system prompt for improving structured outputs
        default_system_prompt = (
            "You are an expert editor that improves structured outputs while maintaining "
            "their original meaning and format. Fix any issues in the output to make it "
            "comply with the specified requirements."
        )

        # Create a prompt critic
        critic = create_prompt_critic(
            llm_provider=model_provider,
            system_prompt=system_prompt or default_system_prompt,
        )

    # Create and return the adapter
    return SifakaPydanticAdapter(
        rules=rules,
        output_model=output_model,
        critic=critic,
        config=adapter_config,
        **kwargs,
    )
