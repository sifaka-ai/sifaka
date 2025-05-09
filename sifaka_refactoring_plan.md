# Sifaka Refactoring Plan: Core Components

## Overview

This document outlines a comprehensive plan to refactor several core components in the Sifaka codebase that currently exist as single files at the root level. The goal is to reorganize these components into proper packages following the established patterns in the codebase.

**Important: This refactoring will NOT maintain backward compatibility.**

Files to be refactored:
1. `/sifaka/generation.py`
2. `/sifaka/improvement.py`
3. `/sifaka/monitoring.py`
4. `/sifaka/validation.py` (already deprecated)

## General Approach

For each component:
1. Create a new package directory
2. Split functionality into appropriate modules
3. Create a proper `__init__.py` that exposes the public API
4. Remove the original file from the root level
5. Update imports throughout the codebase

## 1. Generation Component Refactoring

### Current State
- Single file `/sifaka/generation.py` with `Generator` class
- Simple wrapper around model providers

### Target Structure
```
sifaka/generation/
├── __init__.py
├── generator.py
├── config.py (if needed)
└── interfaces.py (if needed)
```

### Implementation Steps

1. **Create the package structure**
   ```bash
   mkdir -p sifaka/generation
   touch sifaka/generation/__init__.py
   touch sifaka/generation/generator.py
   ```

2. **Move and refactor the Generator class**
   - Move the `Generator` class to `sifaka/generation/generator.py`
   - Enhance with any additional functionality if needed

3. **Create a proper package API in `__init__.py`**
   ```python
   """
   Generation module for Sifaka.

   This package provides components for generating outputs using model providers.
   """

   from .generator import Generator

   __all__ = ["Generator"]
   ```

4. **Remove the original file**
   ```bash
   rm sifaka/generation.py
   ```

5. **Update imports throughout the codebase**
   - Change `from sifaka.generation import Generator` to `from sifaka.generation import Generator`
   - This maintains the same import path but now points to the package

## 2. Improvement Component Refactoring

### Current State
- Single file `/sifaka/improvement.py` with `Improver` class and `ImprovementResult` dataclass
- Handles improving outputs based on validation results

### Target Structure
```
sifaka/improvement/
├── __init__.py
├── improver.py
├── models.py
└── strategies/ (optional for future expansion)
    └── __init__.py
```

### Implementation Steps

1. **Create the package structure**
   ```bash
   mkdir -p sifaka/improvement
   mkdir -p sifaka/improvement/strategies
   touch sifaka/improvement/__init__.py
   touch sifaka/improvement/improver.py
   touch sifaka/improvement/models.py
   touch sifaka/improvement/strategies/__init__.py
   ```

2. **Move and refactor the components**
   - Move `ImprovementResult` to `sifaka/improvement/models.py`
   - Move `Improver` class to `sifaka/improvement/improver.py`
   - Update imports within these files

3. **Create a proper package API in `__init__.py`**
   ```python
   """
   Improvement module for Sifaka.

   This package provides components for improving outputs based on validation results.
   """

   from .improver import Improver
   from .models import ImprovementResult

   __all__ = ["Improver", "ImprovementResult"]
   ```

4. **Remove the original file**
   ```bash
   rm sifaka/improvement.py
   ```

5. **Update imports throughout the codebase**
   - Change `from sifaka.improvement import Improver, ImprovementResult` to `from sifaka.improvement import Improver, ImprovementResult`

## 3. Monitoring Component Refactoring

### Current State
- Single file `/sifaka/monitoring.py` with `TimingStats` and `PerformanceMonitor` classes
- Handles performance monitoring and metrics collection

### Target Structure
```
sifaka/monitoring/
├── __init__.py
├── metrics.py
├── collectors/
│   ├── __init__.py
│   └── timing.py
└── reporters/
    └── __init__.py
```

### Implementation Steps

1. **Create the package structure**
   ```bash
   mkdir -p sifaka/monitoring
   mkdir -p sifaka/monitoring/collectors
   mkdir -p sifaka/monitoring/reporters
   touch sifaka/monitoring/__init__.py
   touch sifaka/monitoring/metrics.py
   touch sifaka/monitoring/collectors/__init__.py
   touch sifaka/monitoring/collectors/timing.py
   touch sifaka/monitoring/reporters/__init__.py
   ```

2. **Move and refactor the components**
   - Move `TimingStats` to `sifaka/monitoring/collectors/timing.py`
   - Move `PerformanceMonitor` to `sifaka/monitoring/metrics.py`
   - Update imports within these files

3. **Create proper package APIs in `__init__.py` files**
   - Main package:
     ```python
     """
     Monitoring module for Sifaka.

     This package provides components for monitoring and analyzing performance metrics.
     """

     from .metrics import PerformanceMonitor
     from .collectors.timing import TimingStats

     __all__ = ["PerformanceMonitor", "TimingStats"]
     ```

   - Collectors subpackage:
     ```python
     """
     Collectors for Sifaka monitoring.

     This subpackage provides collectors for gathering various metrics.
     """

     from .timing import TimingStats

     __all__ = ["TimingStats"]
     ```

4. **Remove the original file**
   ```bash
   rm sifaka/monitoring.py
   ```

5. **Update imports throughout the codebase**
   - Change `from sifaka.monitoring import PerformanceMonitor, TimingStats` to `from sifaka.monitoring import PerformanceMonitor, TimingStats`

## 4. Validation Component (Already Refactored)

The validation component has already been properly refactored into a package structure. The root-level `validation.py` file is currently a deprecated forwarding module.

### Recommendation
- Remove the deprecated file entirely since we're not maintaining backward compatibility
- Update any imports that still use the root-level file

### Implementation Steps

1. **Remove the deprecated file**
   ```bash
   rm sifaka/validation.py
   ```

2. **Update imports throughout the codebase**
   - Change `from sifaka.validation import Validator, ValidationResult` to `from sifaka.validation import Validator, ValidationResult`
   - Or more explicitly: 
     ```python
     from sifaka.validation.models import ValidationResult
     from sifaka.validation.validator import Validator
     ```

## Updating the Main Package Exports

Update the main `sifaka/__init__.py` file to reflect the new package structure:

```python
# Update imports in __init__.py
from sifaka.chain import ChainCore, ChainResult
from sifaka.critics import CriticCore, create_prompt_critic, create_reflexion_critic
from sifaka.generation import Generator
from sifaka.improvement import Improver, ImprovementResult
from sifaka.models import AnthropicProvider, OpenAIProvider
from sifaka.validation import Validator, ValidationResult, ValidatorConfig

# ... rest of the file remains the same
```

## Testing Strategy

After implementing these changes:

1. Run the test suite to identify any broken imports or functionality
2. Fix any issues that arise
3. Ensure documentation is updated to reflect the new structure
4. Verify that all examples in the documentation use the new import paths

## Conclusion

This refactoring will significantly improve the organization and maintainability of the Sifaka codebase by ensuring all major components follow a consistent package structure. By not maintaining backward compatibility, we can make a clean break from the old structure and fully embrace the new organization.
