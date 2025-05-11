# Dependency Management Refinement Plan

This document outlines the plan for refining dependency management in the Sifaka codebase, focusing on addressing circular dependencies and improving dependency injection.

## Current State Analysis

The Sifaka codebase currently has:

1. A dedicated dependency injection module in `sifaka/core/dependency.py` that provides:
   - A `DependencyProvider` singleton class for registering and retrieving dependencies
   - An `inject_dependencies` decorator for injecting dependencies into classes and functions
   - A `DependencyInjector` utility class for manual dependency injection
   - Helper functions like `provide_dependency`, `get_dependency`, etc.

2. Components are instantiated in various ways:
   - Direct instantiation with explicit dependencies passed in constructors
   - Factory functions that create components with dependencies
   - Some components may have hard-coded dependencies

3. Potential circular dependencies between modules, particularly in how components import and use each other.

## Implementation Plan

### 1. Analyze and Resolve Circular Dependencies

#### 1.1 Identify Circular Imports

- Scan the codebase for modules that import each other directly or indirectly
- Focus on core components like Chain, Critic, Rule, etc.
- Examine factory functions and how they import components
- Create a dependency graph to visualize module relationships

#### 1.2 Resolve Circular Imports

- Move interface definitions to dedicated interface modules
- Use type hints with string literals or `TYPE_CHECKING` for forward references
- Implement lazy loading where appropriate
- Restructure imports to avoid circular dependencies

### 2. Standardize Dependency Injection

#### 2.1 Enhance DependencyProvider Implementation

- Add support for scoped dependencies (request, session, singleton)
- Improve error handling and logging
- Add validation for registered dependencies
- Implement dependency resolution strategies

#### 2.2 Enforce Consistent Dependency Injection Patterns

- Use constructor injection as the primary method
- Make all dependencies explicit in constructors
- Avoid creating dependencies inside components
- Use the `inject_dependencies` decorator consistently

### 3. Refactor Factory Functions

#### 3.1 Standardize Factory Function Patterns

- Use consistent parameter naming across all factory functions
- Implement dependency resolution in factory functions
- Add validation for required dependencies
- Use type annotations consistently

#### 3.2 Centralize Component Creation

- Ensure all factory functions use the core dependency injection system
- Implement lazy loading of dependencies in factory functions
- Add support for dependency substitution in tests

### 4. Improve Component Initialization

#### 4.1 Standardize Component Initialization

- Use the `InitializableMixin` consistently
- Implement proper resource management
- Add validation for required dependencies
- Use state management consistently

#### 4.2 Enhance Error Handling

- Add detailed error messages for missing dependencies
- Implement graceful fallbacks where appropriate
- Add validation for dependency compatibility

### 5. Document Dependency Patterns

#### 5.1 Create Documentation

- Document the dependency injection system
- Provide examples of proper dependency usage
- Document component initialization patterns
- Create guidelines for adding new components

## Implementation Approach

### Phase 1: Analysis and Planning

1. Create a dependency graph of the codebase
2. Identify circular dependencies
3. Create a detailed plan for resolving each circular dependency
4. Document current dependency injection patterns

### Phase 2: Core Infrastructure Improvements

1. Enhance the DependencyProvider implementation
2. Add support for scoped dependencies
3. Improve error handling and logging
4. Implement dependency resolution strategies

### Phase 3: Component Refactoring

1. Refactor factory functions to use standardized patterns
2. Update component constructors to use explicit dependency injection
3. Remove hard-coded dependencies
4. Use the `inject_dependencies` decorator consistently

### Phase 4: Documentation and Testing

1. Create documentation for dependency management
2. Provide examples of proper dependency usage
3. Add tests for dependency injection
4. Validate refactored components

## Files to Modify

### Core Infrastructure

1. `sifaka/core/dependency.py`
   - Enhance the DependencyProvider implementation
   - Add support for scoped dependencies
   - Improve error handling and logging

2. `sifaka/core/factories.py`
   - Standardize factory function patterns
   - Implement dependency resolution
   - Add validation for required dependencies

3. `sifaka/core/initialization.py`
   - Enhance the InitializableMixin
   - Standardize component initialization
   - Improve error handling

### Component-Specific Files

1. Chain Component
   - `sifaka/chain/chain.py`
   - `sifaka/chain/factories.py`
   - `sifaka/chain/engine.py`

2. Critic Component
   - `sifaka/critics/core.py`
   - `sifaka/critics/implementations/*.py`
   - `sifaka/critics/factories.py`

3. Rule Component
   - `sifaka/rules/base.py`
   - `sifaka/rules/factories.py`
   - `sifaka/rules/content/*.py`
   - `sifaka/rules/formatting/*.py`

4. Model Component
   - `sifaka/models/core.py`
   - `sifaka/models/providers/*.py`
   - `sifaka/models/factories.py`

5. Retrieval Component
   - `sifaka/retrieval/core.py`
   - `sifaka/retrieval/implementations/*.py`
   - `sifaka/retrieval/factories.py`

### Documentation

1. `docs/dependency_management.md` (new file)
   - Document the dependency injection system
   - Provide examples of proper dependency usage
   - Create guidelines for adding new components

## Success Criteria

1. No circular dependencies in the codebase
2. Consistent dependency injection patterns across all components
3. Improved error handling for missing dependencies
4. Comprehensive documentation for dependency management
5. All components use explicit dependency injection
6. Factory functions follow standardized patterns
7. Component initialization is standardized
8. Tests validate proper dependency injection
