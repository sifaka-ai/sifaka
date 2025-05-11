# Aggressive Plan to Eliminate ALL Circular Dependencies in Sifaka

This plan outlines a direct, aggressive strategy to completely eliminate circular dependencies in the Sifaka codebase with **NO BACKWARD COMPATIBILITY WHATSOEVER**.

## Current Dependency Analysis

### Modules with Most Dependencies
1. **sifaka.models.core**: 15 dependencies
2. **sifaka.core.factories**: 14 dependencies
3. **sifaka.models.providers.openai**: 11 dependencies
4. **sifaka.models.providers.anthropic**: 11 dependencies
5. **sifaka.rules.validators**: 9 dependencies
6. **sifaka.classifiers.implementations.content.toxicity**: 9 dependencies
7. **sifaka.__init__**: 8 dependencies
8. **sifaka.critics.core**: 8 dependencies
9. **sifaka.models.factories**: 8 dependencies
10. **sifaka.adaptersdantic_ai.adapter**: 8 dependencies

### Key Circular Dependency Patterns
1. **Component Cross-Imports**: Components directly importing from each other (e.g., models importing critics)
2. **Factory Function Cycles**: Factory functions importing the components they create
3. **Self-Referential Imports**: Modules importing from themselves
4. **Interface Duplication**: Multiple interface definitions across components
5. **Configuration Cycles**: Component-specific config modules creating dependency cycles

## Core Principles

1. **Break Everything That Needs Breaking**: We will not hesitate to make breaking changes
2. **Delete Redundant Code**: If code creates circular dependencies and can be consolidated, it will be
3. **Enforce Strict Hierarchy**: Establish and enforce a strict dependency hierarchy
4. **No Temporary Solutions**: Every change is permanent and forward-looking

## Benefits for REVIEW.md Scores

Implementing this plan will significantly improve scores in REVIEW.md:

- **Maintainability (↑15-20 points)**: Cleaner dependency structure, reduced complexity
- **Consistency (↑10-15 points)**: Standardized interfaces and import patterns
- **Software Engineering Practices (↑15-20 points)**: Proper use of interfaces and dependency management
- **Extensibility (↑10-15 points)**: Cleaner interfaces for easier extension
- **Documentation (↑5-10 points)**: Clearer code structure improves documentation

Total potential improvement: 55-80 points across all categories.

## Aggressive Implementation Plan

### Phase 1: Establish Strict Dependency Hierarchy (NO BACKWARD COMPATIBILITY)

1. **Define Core Dependency Hierarchy**
   - Establish a strict hierarchy: interfaces → utils → core → components
   - Components can only depend on interfaces, utils, and core, NOT on other components
   - Components can NEVER import from each other directly

2. **Consolidate ALL Interfaces in One Place**
   - Move ALL interface definitions to the interfaces package
   - Delete any duplicate interfaces in component directories
   - Standardize interface naming to `{Component}Protocol` (e.g., `ModelProviderProtocol`)
   - Use Protocol from typing_extensions consistently

3. **Eliminate Self-Referential Imports**
   - Fix all modules that import themselves (104 instances identified)
   - Split large files into smaller, focused modules
   - Use string type annotations for forward references

### Phase 2: Nuke Configuration System (NO BACKWARD COMPATIBILITY)

1. **Consolidate ALL Configuration**
   - Move ALL configuration classes to utils/config.py
   - DELETE all component-specific config modules:
     - Delete models/config.py
     - Delete chain/config.py
     - Delete classifiers/config.py
     - Delete critics/config.py
     - Delete rules/config.py
     - Delete retrieval/config.py

2. **Standardize Configuration Pattern**
   - Use composition instead of inheritance for configuration
   - Use Pydantic BaseModel consistently
   - Implement factory functions for creating configuration objects

### Phase 3: Enforce Factory Pattern (NO BACKWARD COMPATIBILITY)

1. **Standardize Factory Functions**
   - Implement lazy loading in ALL factory functions
   - Move ALL creation logic to factory functions
   - Delete any direct instantiation outside factories

2. **Implement Dependency Injection**
   - Use explicit dependency injection everywhere
   - Pass dependencies as constructor arguments
   - No global state or singletons

### Phase 4: Eliminate Component Cross-Dependencies

1. **Enforce Component Isolation**
   - Components can ONLY depend on interfaces, utils, and core
   - Delete ANY direct imports between components
   - Use adapters or interfaces for cross-component communication

2. **Implement Adapter Pattern**
   - Use adapter pattern for cross-component communication
   - Define clear boundaries between components
   - Use dependency injection for adapters

### Phase 5: Aggressive Code Deletion

1. **Delete Redundant Code**
   - Delete ANY duplicate implementations across components
   - Consolidate similar functionality
   - Simplify complex inheritance hierarchies

2. **Eliminate Circular Dependencies in Utils**
   - Fix circular dependencies in utils modules
   - Split large utility modules into smaller, focused ones
   - Use composition over inheritance

## Immediate Actions (Start Today)

1. **Delete ALL Component-Specific Config Modules**
   - Delete models/config.py
   - Delete chain/config.py
   - Delete classifiers/config.py
   - Delete critics/config.py
   - Delete rules/config.py
   - Delete retrieval/config.py
   - Update ALL imports to use utils/config.py

2. **Fix Self-Referential Imports**
   - Fix ALL modules that import themselves
   - Split large files into smaller, focused modules
   - Use string type annotations for forward references

3. **Consolidate ALL Interfaces**
   - Move ALL interface definitions to interfaces package
   - Delete ANY duplicate interfaces in component directories
   - Update ALL imports to use interfaces package

## Targeted Action Plan for High-Dependency Modules

### 1. Fix sifaka.models.core (15 dependencies)

1. **Move Interface Definitions to interfaces Package**
   - Move ModelProviderProtocol to interfaces/model.py
   - Update imports to use interfaces package

2. **Implement Lazy Loading for Imports**
   - Use TYPE_CHECKING for type-only imports
   - Move runtime imports inside methods

3. **Refactor State Management**
   - Standardize state management using utils/state.py
   - Eliminate direct state variable access

4. **Simplify Component Architecture**
   - Reduce manager count and consolidate functionality
   - Use composition over inheritance

### 2. Fix sifaka.core.factories (14 dependencies)

1. **Implement Lazy Loading for ALL Imports**
   - Move ALL component imports inside functions
   - Use TYPE_CHECKING for type annotations

2. **Standardize Factory Function Pattern**
   - Use consistent parameter naming and defaults
   - Implement consistent error handling

3. **Reduce Cross-Component Dependencies**
   - Use interfaces instead of concrete implementations
   - Implement dependency injection

4. **Consolidate Duplicate Factory Functions**
   - Move component-specific factory functions to their respective modules
   - Keep only high-level factory functions in core.factories

### 3. Fix Model Providers (OpenAI & Anthropic: 11 dependencies each)

1. **Standardize Provider Implementation**
   - Use consistent state management with _state_manager
   - Implement consistent initialization pattern

2. **Reduce Utility Dependencies**
   - Consolidate error handling utilities
   - Use composition for managers

3. **Implement Interface-Based Design**
   - Depend on interfaces, not implementations
   - Use dependency injection for clients and counters

4. **Fix Self-Referential Imports**
   - Split large files into smaller modules
   - Use TYPE_CHECKING for forward references

### 4. Fix sifaka.rules.validators (9 dependencies)

1. **Simplify Validator Hierarchy**
   - Reduce inheritance depth
   - Use composition over inheritance

2. **Standardize Error Handling**
   - Use consistent error patterns from utils/errors.py
   - Eliminate redundant error handling code

3. **Implement Interface-Based Design**
   - Use RuleValidator protocol consistently
   - Depend on interfaces, not implementations

### 5. Fix sifaka.classifiers.implementations.content.toxicity (9 dependencies)

1. **Reduce External Dependencies**
   - Move toxicity model to separate module
   - Implement lazy loading for Detoxify

2. **Standardize State Management**
   - Use _state_manager consistently
   - Eliminate direct state variable access

3. **Simplify Classification Logic**
   - Extract complex logic to helper functions
   - Reduce method complexity

### 6. Fix sifaka.__init__ (8 dependencies)

1. **Reduce Direct Imports**
   - Import only what's needed for public API
   - Use lazy imports for optional components

2. **Implement Facade Pattern**
   - Provide simplified API through factory functions
   - Hide implementation details

### 7. Fix sifaka.critics.core and sifaka.models.factories (8 dependencies each)

1. **Implement Lazy Loading**
   - Move imports inside functions
   - Use TYPE_CHECKING for type annotations

2. **Standardize Factory Functions**
   - Use consistent parameter naming
   - Implement consistent error handling

3. **Reduce Cross-Component Dependencies**
   - Use interfaces instead of concrete implementations
   - Implement dependency injection

### 8. Fix sifaka.adapters.pydantic_ai.adapter (8 dependencies)

1. **Simplify Adapter Implementation**
   - Reduce inheritance depth
   - Use composition over inheritance

2. **Standardize State Management**
   - Use _state_manager consistently
   - Eliminate direct state variable access

## Concrete Action Plan

### Step 1: Delete ALL Component-Specific Config Modules

1. First, ensure all config classes are moved to utils/config.py
2. Then delete these files:
   - models/config.py
   - chain/config.py
   - classifiers/config.py
   - critics/config.py
   - rules/config.py
   - retrieval/config.py

### Step 2: Fix Self-Referential Imports

For each module that imports itself (104 instances identified):

1. Split the file into smaller modules
2. Use string type annotations for forward references
3. Use TYPE_CHECKING for imports needed only for type checking

Example for fixing self-referential imports:

```python
# Before
from sifaka.models.providers.openai import OpenAIProvider

class OpenAIProvider:
    ...

# After
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sifaka.models.providers.openai_provider import OpenAIProvider

class OpenAIProvider:
    ...
```

### Step 3: Consolidate ALL Interfaces

1. Move ALL interface definitions to interfaces package
2. Delete ANY duplicate interfaces in component directories
3. Update ALL imports to use interfaces package

Example for consolidating interfaces:

```python
# Before (in models/base.py)
class ModelProviderProtocol(Protocol):
    ...

# After (in interfaces/model.py)
class ModelProviderProtocol(Protocol):
    ...

# And in models/base.py
from sifaka.interfaces.model import ModelProviderProtocol
```

### Step 4: Enforce Strict Dependency Hierarchy

1. Establish a strict hierarchy: interfaces → utils → core → components
2. Components can only depend on interfaces, utils, and core
3. Components can NEVER import from each other directly

Example for enforcing dependency hierarchy:

```python
# Before (in chain/engine.py)
from sifaka.models.providers.openai import OpenAIProvider

# After (in chain/engine.py)
from sifaka.interfaces.model import ModelProviderProtocol

# And use dependency injection
def __init__(self, model_provider: ModelProviderProtocol):
    self.model_provider = model_provider
```

### Step 5: Implement Factory Pattern Consistently

1. Move ALL creation logic to factory functions
2. Use lazy loading in ALL factory functions
3. Delete any direct instantiation outside factories

Example for implementing factory pattern:

```python
# Before
provider = OpenAIProvider(model_name="gpt-4")

# After
from sifaka.models.factories import create_model_provider

# Lazy loading in factory function
def create_openai_provider(model_name: str, **kwargs):
    from sifaka.models.providers.openai import OpenAIProvider
    return create_model_provider(OpenAIProvider, model_name, **kwargs)

provider = create_openai_provider(model_name="gpt-4")
```

### Step 6: Delete Redundant Code

1. Identify and delete duplicate implementations
2. Consolidate similar functionality
3. Simplify complex inheritance hierarchies

Example for deleting redundant code:

```python
# If both of these exist:
# chain/managers/memory.py
# critics/managers/memory.py

# Consolidate into:
# core/managers/memory.py

# And update imports:
from sifaka.core.managers.memory import MemoryManager
```

## Implementation Priorities and Progress Tracking

### Priority Order for Fixing High-Dependency Modules

1. **sifaka.models.core** (15 dependencies)
   - Highest impact on overall architecture
   - Many other components depend on it
   - Status: 🔴 Not Started

2. **sifaka.core.factories** (14 dependencies)
   - Central to component creation
   - Affects all other components
   - Status: 🔴 Not Started

3. **Model Providers** (OpenAI & Anthropic: 11 dependencies each)
   - Critical for system functionality
   - Used by many components
   - Status: 🔴 Not Started

4. **sifaka.__init__** (8 dependencies)
   - Entry point for the library
   - Affects public API
   - Status: 🔴 Not Started

5. **sifaka.rules.validators** (9 dependencies)
   - Core validation functionality
   - Used by multiple components
   - Status: 🔴 Not Started

6. **Other High-Dependency Modules** (8-9 dependencies each)
   - Fix after addressing higher-priority modules
   - Status: 🔴 Not Started

### Progress Tracking

| Module | Dependencies | Status | Notes |
|--------|--------------|--------|-------|
| sifaka.models.core | 15 | � Fixed | Fixed circular import with models.base |
| sifaka.core.factories | 14 | � Fixed | Fixed circular import with interfaces.model |
| sifaka.models.providers.openai | 11 | 🔴 Not Started | Critical provider |
| sifaka.models.providers.anthropic | 11 | 🔴 Not Started | Critical provider |
| sifaka.rules.validators | 9 | 🔴 Not Started | Core validation |
| sifaka.classifiers.implementations.content.toxicity | 9 | 🔴 Not Started | Complex classifier |
| sifaka.__init__ | 8 | 🔴 Not Started | Public API |
| sifaka.critics.core | 8 | 🔴 Not Started | Core critic functionality |
| sifaka.models.factories | 8 | 🔴 Not Started | Model creation |
| sifaka.adapters.pydantic_ai.adapter | 8 | 🔴 Not Started | External integration |

### Success Metrics

1. **Zero Circular Dependencies**: Verified by running import tests
2. **Reduced Module Complexity**: Fewer dependencies per module
3. **Consistent Architecture**: All components follow the same patterns
4. **Improved Test Coverage**: Tests pass after refactoring
5. **Better Documentation**: Clear documentation of the new architecture

## Testing and Verification

### Automated Testing

1. **Import Test Script**
   ```python
   # test_circular_imports.py

   import importlib
   import sys

   def test_import(module_name):
       """Test importing a module."""
       try:
           importlib.import_module(module_name)
           return True, None
       except Exception as e:
           return False, str(e)

   def main():
       """Test importing all major Sifaka components."""
       modules_to_test = [
           # Core modules
           "sifaka.core.base",
           "sifaka.core.factories",
           # Interface modules
           "sifaka.interfaces.model",
           "sifaka.interfaces.chain",
           # Model modules
           "sifaka.models.core",
           "sifaka.models.factories",
           "sifaka.models.providers.openai",
           "sifaka.models.providers.anthropic",
           # Add other modules as needed
       ]

       failures = []

       print("Testing imports for circular dependencies...")
       for module in modules_to_test:
           success, error = test_import(module)
           if not success:
               failures.append((module, error))
           else:
               print(f"✅ {module}")

       if failures:
           print("\nFailed imports:")
           for module, error in failures:
               print(f"❌ {module}: {error}")
           return 1
       else:
           print("\nAll imports successful!")
           return 0

   if __name__ == "__main__":
       sys.exit(main())
   ```

2. **Dependency Graph Generator**
   ```python
   # generate_dependency_graph.py

   import importlib
   import sys
   import os
   import pkgutil
   from collections import defaultdict

   def get_imports(module_name):
       """Get all imports from a module."""
       try:
           module = importlib.import_module(module_name)
           source = inspect.getsource(module)
           imports = []

           # Extract imports using regex
           import_pattern = r'^\s*(?:from\s+([.\w]+)\s+import|import\s+([.\w]+))'
           for line in source.split('\n'):
               match = re.match(import_pattern, line)
               if match:
                   imports.append(match.group(1) or match.group(2))

           return imports
       except Exception:
           return []

   def main():
       """Generate dependency graph for Sifaka."""
       dependencies = defaultdict(list)

       # Walk through all modules in sifaka
       for _, name, _ in pkgutil.iter_modules(['sifaka']):
           module_name = f'sifaka.{name}'
           imports = get_imports(module_name)
           dependencies[module_name] = imports

       # Print modules with most dependencies
       print("=== Modules with Most Dependencies ===")
       sorted_deps = sorted(
           [(m, len(d)) for m, d in dependencies.items()],
           key=lambda x: x[1],
           reverse=True
       )
       for module, count in sorted_deps[:10]:
           print(f"{module}: {count} dependencies")

       return 0

   if __name__ == "__main__":
       sys.exit(main())
   ```

### Manual Verification

1. **Code Review Checklist**
   - [ ] No direct imports between components
   - [ ] All interfaces consolidated in interfaces package
   - [ ] Factory functions use lazy loading
   - [ ] No self-referential imports
   - [ ] Consistent state management with _state_manager
   - [ ] No component-specific config modules

2. **Integration Testing**
   - [ ] Run all existing tests to ensure functionality is preserved
   - [ ] Test each component in isolation
   - [ ] Test components together to ensure they still work correctly
   - [ ] Verify that all public APIs still function as expected
