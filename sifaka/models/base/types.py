"""
Model Types Module

This module provides type definitions and aliases for the model system.

## Overview
The types module defines type variables and aliases used throughout the model system.
It centralizes type definitions to ensure consistency and avoid circular imports.

## Components
- **Type Variables**: Generic type variables for model providers and configurations

## Usage Examples
```python
from sifaka.models.base.types import T, C

# Use type variables in generic classes
class CustomProvider(Generic[C]):
    def __init__(self, config: C):
        self.config = config
```
"""

from typing import TypeVar

# No need to import LanguageModelProtocol anymore

# Type variables for generic type definitions
T = TypeVar("T")  # Generic type for ModelProvider
C = TypeVar("C", bound="ModelConfig")  # Config type

# Import here to avoid circular import in the type annotation above
from sifaka.utils.config.models import ModelConfig  # noqa: E402
