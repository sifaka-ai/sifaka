"""Input validation helpers for Sifaka.

This module provides validation functions for common input parameters
used throughout the Sifaka codebase. These functions ensure that
inputs meet requirements and provide clear error messages.

Functions:
- validate_prompt: Validate text prompts
- validate_max_iterations: Validate iteration limits
- validate_model_name: Validate model identifiers
- validate_timeout: Validate timeout values
"""

import re
from typing import Optional

from sifaka.utils.errors import ValidationError


def validate_prompt(prompt: str, min_length: int = 1, max_length: int = 10000) -> str:
    """Validate a text prompt.
    
    Args:
        prompt: The prompt text to validate
        min_length: Minimum allowed length (default: 1)
        max_length: Maximum allowed length (default: 10000)
        
    Returns:
        The validated prompt (stripped of whitespace)
        
    Raises:
        ValidationError: If prompt is invalid
        
    Example:
        ```python
        prompt = validate_prompt("  Write about AI  ")
        # Returns: "Write about AI"
        ```
    """
    if not isinstance(prompt, str):
        raise ValidationError(
            f"Prompt must be a string, got {type(prompt).__name__}",
            validator_name="prompt_type",
            validation_details={"expected_type": "str", "actual_type": type(prompt).__name__},
            suggestions=[
                "Ensure prompt is a string",
                "Convert input to string if needed",
            ]
        )
    
    # Strip whitespace
    prompt = prompt.strip()
    
    if not prompt:
        raise ValidationError(
            "Prompt cannot be empty or only whitespace",
            validator_name="prompt_empty",
            validation_details={"original_prompt": repr(prompt)},
            suggestions=[
                "Provide a non-empty prompt",
                "Check that prompt contains meaningful text",
            ]
        )
    
    if len(prompt) < min_length:
        raise ValidationError(
            f"Prompt too short: {len(prompt)} characters (minimum: {min_length})",
            validator_name="prompt_length",
            validation_details={
                "length": len(prompt),
                "min_length": min_length,
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt
            },
            suggestions=[
                f"Provide a prompt with at least {min_length} characters",
                "Add more detail to your prompt",
            ]
        )
    
    if len(prompt) > max_length:
        raise ValidationError(
            f"Prompt too long: {len(prompt)} characters (maximum: {max_length})",
            validator_name="prompt_length",
            validation_details={
                "length": len(prompt),
                "max_length": max_length,
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt
            },
            suggestions=[
                f"Shorten prompt to {max_length} characters or less",
                "Break long prompts into multiple thoughts",
                "Focus on the most important aspects",
            ]
        )
    
    return prompt


def validate_max_iterations(max_iterations: int, min_value: int = 1, max_value: int = 20) -> int:
    """Validate maximum iterations parameter.
    
    Args:
        max_iterations: The maximum iterations value to validate
        min_value: Minimum allowed value (default: 1)
        max_value: Maximum allowed value (default: 20)
        
    Returns:
        The validated max_iterations value
        
    Raises:
        ValidationError: If max_iterations is invalid
        
    Example:
        ```python
        iterations = validate_max_iterations(5)
        # Returns: 5
        ```
    """
    if not isinstance(max_iterations, int):
        raise ValidationError(
            f"max_iterations must be an integer, got {type(max_iterations).__name__}",
            validator_name="max_iterations_type",
            validation_details={
                "expected_type": "int",
                "actual_type": type(max_iterations).__name__,
                "value": max_iterations
            },
            suggestions=[
                "Ensure max_iterations is an integer",
                "Convert to integer if needed: int(value)",
            ]
        )
    
    if max_iterations < min_value:
        raise ValidationError(
            f"max_iterations too small: {max_iterations} (minimum: {min_value})",
            validator_name="max_iterations_range",
            validation_details={
                "value": max_iterations,
                "min_value": min_value,
                "max_value": max_value
            },
            suggestions=[
                f"Set max_iterations to at least {min_value}",
                "Use 3 as a reasonable default for most use cases",
            ]
        )
    
    if max_iterations > max_value:
        raise ValidationError(
            f"max_iterations too large: {max_iterations} (maximum: {max_value})",
            validator_name="max_iterations_range",
            validation_details={
                "value": max_iterations,
                "min_value": min_value,
                "max_value": max_value
            },
            suggestions=[
                f"Set max_iterations to {max_value} or less",
                "Consider if your use case really needs many iterations",
                "Most tasks complete successfully in 3-5 iterations",
            ]
        )
    
    return max_iterations


def validate_model_name(model_name: str) -> str:
    """Validate a model name/identifier.
    
    Args:
        model_name: The model name to validate
        
    Returns:
        The validated model name
        
    Raises:
        ValidationError: If model name is invalid
        
    Example:
        ```python
        model = validate_model_name("openai:gpt-4")
        # Returns: "openai:gpt-4"
        ```
    """
    if not isinstance(model_name, str):
        raise ValidationError(
            f"Model name must be a string, got {type(model_name).__name__}",
            validator_name="model_name_type",
            validation_details={
                "expected_type": "str",
                "actual_type": type(model_name).__name__,
                "value": model_name
            },
            suggestions=[
                "Ensure model name is a string",
                "Use format: 'provider:model-name'",
            ]
        )
    
    model_name = model_name.strip()
    
    if not model_name:
        raise ValidationError(
            "Model name cannot be empty",
            validator_name="model_name_empty",
            suggestions=[
                "Provide a valid model name",
                "Use format: 'provider:model-name' (e.g., 'openai:gpt-4')",
            ]
        )
    
    # Basic format validation (provider:model)
    if ":" not in model_name:
        raise ValidationError(
            f"Invalid model name format: {model_name}",
            validator_name="model_name_format",
            validation_details={"model_name": model_name},
            suggestions=[
                "Use format: 'provider:model-name'",
                "Examples: 'openai:gpt-4', 'anthropic:claude-3-sonnet'",
            ]
        )
    
    provider, model = model_name.split(":", 1)
    
    if not provider:
        raise ValidationError(
            f"Model provider cannot be empty: {model_name}",
            validator_name="model_provider_empty",
            validation_details={"model_name": model_name},
            suggestions=[
                "Specify a provider before the colon",
                "Examples: 'openai:gpt-4', 'anthropic:claude-3-sonnet'",
            ]
        )
    
    if not model:
        raise ValidationError(
            f"Model name cannot be empty after provider: {model_name}",
            validator_name="model_name_empty",
            validation_details={"model_name": model_name},
            suggestions=[
                "Specify a model name after the colon",
                "Examples: 'openai:gpt-4', 'anthropic:claude-3-sonnet'",
            ]
        )
    
    return model_name


def validate_timeout(timeout: float, min_value: float = 0.1, max_value: float = 300.0) -> float:
    """Validate a timeout value.
    
    Args:
        timeout: The timeout value to validate (in seconds)
        min_value: Minimum allowed value (default: 0.1)
        max_value: Maximum allowed value (default: 300.0)
        
    Returns:
        The validated timeout value
        
    Raises:
        ValidationError: If timeout is invalid
        
    Example:
        ```python
        timeout = validate_timeout(30.0)
        # Returns: 30.0
        ```
    """
    if not isinstance(timeout, (int, float)):
        raise ValidationError(
            f"Timeout must be a number, got {type(timeout).__name__}",
            validator_name="timeout_type",
            validation_details={
                "expected_type": "float or int",
                "actual_type": type(timeout).__name__,
                "value": timeout
            },
            suggestions=[
                "Ensure timeout is a number (int or float)",
                "Use seconds as the unit (e.g., 30.0 for 30 seconds)",
            ]
        )
    
    timeout = float(timeout)
    
    if timeout < min_value:
        raise ValidationError(
            f"Timeout too small: {timeout} seconds (minimum: {min_value})",
            validator_name="timeout_range",
            validation_details={
                "value": timeout,
                "min_value": min_value,
                "max_value": max_value
            },
            suggestions=[
                f"Set timeout to at least {min_value} seconds",
                "Use 30.0 seconds as a reasonable default",
            ]
        )
    
    if timeout > max_value:
        raise ValidationError(
            f"Timeout too large: {timeout} seconds (maximum: {max_value})",
            validator_name="timeout_range",
            validation_details={
                "value": timeout,
                "min_value": min_value,
                "max_value": max_value
            },
            suggestions=[
                f"Set timeout to {max_value} seconds or less",
                "Consider if such a long timeout is really needed",
                "Most API calls complete within 30-60 seconds",
            ]
        )
    
    return timeout
