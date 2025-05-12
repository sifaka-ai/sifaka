# Config Module Refactoring Plan

## Overview

This document outlines the detailed plan for refactoring the `sifaka/utils/config.py` file (2,810 lines) into a more maintainable directory structure with smaller, focused modules.

## Current State

- Single file containing all configuration classes and standardization functions
- Imported by approximately 50 files across the codebase
- Contains configuration classes for all major components (models, rules, critics, chain, classifiers, retrieval)
- Contains standardization functions for each component type

## Target Structure

```
sifaka/utils/config/
├── __init__.py         # Exports and standardization functions
├── base.py             # BaseConfig class
├── models.py           # Model configurations (ModelConfig, OpenAIConfig, etc.)
├── rules.py            # Rule configurations
├── critics.py          # Critic configurations
├── chain.py            # Chain configurations
├── classifiers.py      # Classifier configurations
└── retrieval.py        # Retrieval configurations
```

## Implementation Steps

### 1. Create Directory Structure

```bash
mkdir -p sifaka/utils/config
touch sifaka/utils/config/__init__.py
touch sifaka/utils/config/base.py
touch sifaka/utils/config/models.py
touch sifaka/utils/config/rules.py
touch sifaka/utils/config/critics.py
touch sifaka/utils/config/chain.py
touch sifaka/utils/config/classifiers.py
touch sifaka/utils/config/retrieval.py
```

### 2. Implement Base Module

**File: sifaka/utils/config/base.py**

Content:
- Import statements
- BaseConfig class
- Common utility functions

### 3. Implement Component-Specific Modules

For each component type (models, rules, critics, chain, classifiers, retrieval):
1. Extract relevant configuration classes
2. Extract relevant standardization functions
3. Update imports to reference base.py

### 4. Implement __init__.py

**File: sifaka/utils/config/__init__.py**

Content:
- Re-export all classes and functions to maintain backward compatibility
- Import all configuration classes from component-specific modules
- Import all standardization functions from component-specific modules

### 5. Update Imports Throughout Codebase

- No changes needed if backward compatibility is maintained through __init__.py

### 6. Testing

- Run unit tests for configuration classes
- Run integration tests for components that use configuration
- Verify that all imports work correctly

## Detailed Module Contents

### base.py

```python
"""
Base Configuration Module

This module provides the base configuration class for all Sifaka components.
"""

from typing import Any, Dict, TypeVar, Generic
from pydantic import BaseModel, Field, ConfigDict

# Type variables for generic configuration handling
T = TypeVar("T", bound=BaseModel)

class BaseConfig(BaseModel):
    """Base configuration for all Sifaka components."""
    name: str = Field(default="", description="Component name")
    description: str = Field(default="", description="Component description")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

    model_config = ConfigDict(frozen=True)

    def with_params(self, **kwargs: Any) -> "BaseConfig":
        """Create a new configuration with updated parameters."""
        return self.model_copy(update={"params": {**self.params, **kwargs}})

    def with_options(self, **kwargs: Any) -> "BaseConfig":
        """Create a new configuration with updated options."""
        return self.model_copy(update=kwargs)
```

### models.py

```python
"""
Model Configuration Module

This module provides configuration classes for model providers.
"""

from typing import Any, Dict, Optional, Type, TypeVar, Union, cast
from pydantic import Field

from .base import BaseConfig

# Configuration classes
class ModelConfig(BaseConfig):
    """Configuration for model providers."""
    model: str = Field(default="", description="Model name to use")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Temperature for text generation")
    max_tokens: int = Field(default=1000, ge=1, description="Maximum number of tokens to generate")
    api_key: Optional[str] = Field(default=None, description="API key for the model provider")
    trace_enabled: bool = Field(default=False, description="Whether to enable tracing")

class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI model providers."""
    pass

class AnthropicConfig(ModelConfig):
    """Configuration for Anthropic model providers."""
    pass

class GeminiConfig(ModelConfig):
    """Configuration for Google Gemini model providers."""
    pass

# Standardization functions
def standardize_model_config(
    config: Optional[Union[Dict[str, Any], ModelConfig]] = None,
    params: Optional[Dict[str, Any]] = None,
    config_class: Type[T] = ModelConfig,
    **kwargs: Any,
) -> T:
    """Standardize model provider configuration."""
    # Implementation...
```

### critics.py, rules.py, chain.py, classifiers.py, retrieval.py

Similar structure to models.py, with component-specific configuration classes and standardization functions.

### __init__.py

```python
"""
Configuration Utilities

This module provides a unified configuration system for the Sifaka framework,
including base configuration classes and standardization functions for different
component types.
"""

# Re-export everything from base
from .base import *

# Re-export from component-specific modules
from .models import *
from .rules import *
from .critics import *
from .chain import *
from .classifiers import *
from .retrieval import *
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing imports | Maintain backward compatibility through __init__.py |
| Circular dependencies | Carefully organize imports to avoid circular references |
| Missing exports | Comprehensive testing to verify all required exports |
| Performance impact | Profile before and after to ensure no significant changes |

## Testing Strategy

1. **Unit Tests**: Verify each configuration class works correctly
2. **Integration Tests**: Verify components can use configuration classes
3. **Import Tests**: Verify all imports work correctly
4. **Performance Tests**: Verify no significant performance impact

## Timeline

1. **Setup**: Create directory structure and empty files (0.5 day)
2. **Implementation**: Implement each module (2 days)
3. **Testing**: Comprehensive testing (1 day)
4. **Documentation**: Update documentation (0.5 day)

Total: 4 days
