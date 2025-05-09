# Implementation Plan for Reorganizing Sifaka Interfaces

## Overview

This document outlines the plan for reorganizing the interfaces directory in the Sifaka project to match the structure of other components like `chain`, `critics`, and `rules`. The goal is to improve modularity, consistency, and maintainability of the codebase.

## What Interfaces Do in Sifaka

Interfaces in Sifaka serve several important purposes:

1. **Define Contracts**: Interfaces establish clear contracts that components must fulfill, ensuring consistent behavior across different implementations.

2. **Enable Structural Subtyping**: Sifaka uses Python's `Protocol` class to define interfaces, enabling structural subtyping where classes don't need to explicitly inherit from interfaces.

3. **Promote Modularity**: By defining clear interfaces, components can be swapped out without affecting the rest of the system.

4. **Facilitate Testing**: Interfaces make it easier to create mock implementations for testing.

5. **Improve Documentation**: Interfaces serve as documentation for how components should behave.

## Current Structure

The current structure has a top-level `interfaces` directory with:
- Core interfaces in `core.py` (Component, Configurable, Stateful, Identifiable, etc.)
- Domain-specific interfaces in separate files (models.py, critics.py, rules.py, chain.py)

## Target Structure

The target structure will distribute interfaces to their respective component directories:

```
sifaka/
├── core/
│   ├── __init__.py
│   └── interfaces.py        # Core interfaces (Component, Configurable, etc.)
├── models/
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── provider.py      # ModelProvider interface
│   │   ├── client.py        # APIClient interface
│   │   └── counter.py       # TokenCounter interface
├── rules/
│   ├── interfaces/
│   │   ├── __init__.py
│   │   └── rule.py          # Rule interface
├── critics/
│   ├── interfaces/
│   │   ├── __init__.py
│   │   └── critic.py        # Critic interface
```

## Implementation Phases

### Phase 1: Create New Interface Files (Week 1)

1. **Create Core Module**:
   - Create `sifaka/core/` directory
   - Create `sifaka/core/__init__.py`
   - Create `sifaka/core/interfaces.py` with core interfaces from `interfaces/core.py`

2. **Create Models Interfaces**:
   - Ensure `sifaka/models/interfaces/` directory exists
   - Create `provider.py`, `client.py`, `counter.py` with interfaces from `interfaces/models.py`
   - Update `sifaka/models/interfaces/__init__.py` to export these interfaces

3. **Create Rules Interfaces**:
   - Ensure `sifaka/rules/interfaces/` directory exists
   - Create `rule.py` with interfaces from `interfaces/rules.py`
   - Update `sifaka/rules/interfaces/__init__.py` to export these interfaces

4. **Create Critics Interfaces**:
   - Ensure `sifaka/critics/interfaces/` directory exists
   - Create `critic.py` with interfaces from `interfaces/critics.py`
   - Update `sifaka/critics/interfaces/__init__.py` to export these interfaces

### Phase 2: Update Import Statements (Week 2)

1. **Update Core Component Imports**:
   - Update imports in core components to use the new interface locations
   - For example, change `from sifaka.interfaces.core import Component` to `from sifaka.core.interfaces import Component`

2. **Update Models Component Imports**:
   - Update imports in models components to use the new interface locations
   - For example, change `from sifaka.interfaces.models import ModelProvider` to `from sifaka.models.interfaces.provider import ModelProvider`

3. **Update Rules Component Imports**:
   - Update imports in rules components to use the new interface locations
   - For example, change `from sifaka.interfaces.rules import Rule` to `from sifaka.rules.interfaces.rule import Rule`

4. **Update Critics Component Imports**:
   - Update imports in critics components to use the new interface locations
   - For example, change `from sifaka.interfaces.critics import Critic` to `from sifaka.critics.interfaces.critic import Critic`

5. **Update Chain Component Imports**:
   - Update imports in chain components to use the new interface locations
   - For example, change `from sifaka.interfaces.chain import Chain` to `from sifaka.chain.interfaces.chain import Chain`

### Phase 3: DO NOT Maintain Backward Compatibility (Week 2)

DELETE EVERYTHING WE NOT LONGER NEED

### Phase 4: Testing and Validation (Week 3)

1. **Run All Tests**:
   - Run all unit tests to ensure the reorganization hasn't broken anything
   - Fix any issues that arise

2. **Manual Testing**:
   - Manually test key functionality to ensure everything works as expected
   - Check that all components can be instantiated and used correctly

3. **Documentation Updates**:
   - Update docstrings to reflect the new organization
   - Update any documentation that references the old interface locations

### Phase 5: Cleanup (Week 4)

1. **Remove Duplicate Code**:
   - Once all tests pass, remove duplicate code from the original interface files
   - Keep only the re-exports and deprecation warnings

2. **Final Review**:
   - Conduct a final review of the codebase to ensure all interfaces are properly organized
   - Check for any missed imports or references to the old locations

## Detailed File Changes

### Core Interfaces

```python
# sifaka/core/interfaces.py
"""
Core interfaces for Sifaka.

This module defines the fundamental interfaces that all components in the Sifaka
framework should implement. These interfaces establish a common contract for
component behavior, enabling better modularity and extensibility.
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar, runtime_checkable

# Type variables
T = TypeVar("T")
ConfigType = TypeVar("ConfigType")
StateType = TypeVar("StateType")

@runtime_checkable
class Component(Protocol):
    """Base interface for all components in Sifaka."""
    # Implementation from interfaces/core.py
    ...

@runtime_checkable
class Configurable(Protocol[ConfigType]):
    """Interface for components with configuration."""
    # Implementation from interfaces/core.py
    ...

# Additional core interfaces...
```

### Models Interfaces

```python
# sifaka/models/interfaces/provider.py
"""
Model provider interfaces for Sifaka.

This module defines the interfaces for model providers in the Sifaka framework.
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, runtime_checkable

from sifaka.core.interfaces import Component, Configurable, Identifiable, Stateful

# Type variables
T = TypeVar("T")
ConfigType = TypeVar("ConfigType")
ModelConfigType = TypeVar("ModelConfigType")
PromptType = TypeVar("PromptType")
ResponseType = TypeVar("ResponseType")

@runtime_checkable
class ModelProvider(Protocol):
    """Interface for model providers."""
    # Implementation from interfaces/models.py
    ...

# Additional model provider interfaces...
```

### Backward Compatibility

```python
# sifaka/interfaces/__init__.py
"""
Interfaces module for Sifaka.

This module provides the core interfaces and protocols that define the contracts
for all components in the Sifaka framework.

Note: This module re-exports interfaces from their new locations for backward compatibility.
New code should import interfaces directly from their component-specific locations.
"""

import warnings

warnings.warn(
    "Importing from sifaka.interfaces is deprecated. "
    "Please import interfaces from their component-specific locations, "
    "e.g., from sifaka.core.interfaces, sifaka.models.interfaces, etc.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export core interfaces
from sifaka.core.interfaces import (
    Component,
    Configurable,
    Stateful,
    Identifiable,
    Loggable,
    Traceable,
)

# Re-export model interfaces
from sifaka.models.interfaces.provider import ModelProvider, AsyncModelProvider
from sifaka.models.interfaces.client import APIClient
from sifaka.models.interfaces.counter import TokenCounter

# Additional re-exports...

__all__ = [
    # Core interfaces
    "Component",
    "Configurable",
    "Stateful",
    "Identifiable",
    "Loggable",
    "Traceable",
    # Model interfaces
    "ModelProvider",
    "AsyncModelProvider",
    "APIClient",
    "TokenCounter",
    # Additional interfaces...
]
```

## Risks and Mitigations

1. **Risk**: Breaking existing code that imports from the original locations.
   **Mitigation**: Maintain backward compatibility through re-exports and add deprecation warnings.

2. **Risk**: Missing some import statements during the update.
   **Mitigation**: Run comprehensive tests and conduct thorough code reviews.

3. **Risk**: Introducing bugs during the reorganization.
   **Mitigation**: Implement changes incrementally and test after each phase.

## Success Criteria

1. All interfaces are properly organized in their respective component directories.
2. All import statements are updated to use the new interface locations.
3. Backward compatibility is maintained through re-exports.
4. All tests pass after the reorganization.
5. Documentation is updated to reflect the new organization.

## Timeline

- **Week 1**: Create new interface files
- **Week 2**: Update import statements and maintain backward compatibility
- **Week 3**: Testing and validation
- **Week 4**: Cleanup and final review

## Conclusion

This reorganization will improve the modularity, consistency, and maintainability of the Sifaka codebase. By distributing interfaces to their respective component directories, we'll make the codebase more intuitive and easier to navigate. The phased approach will minimize disruption and ensure a smooth transition.
