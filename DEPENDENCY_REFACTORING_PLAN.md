# Dependency Injection Refactoring Plan

This document outlines the plan for refactoring the `core/dependency.py` file into a modular directory structure.

## Current State

The current `core/dependency.py` file is approximately 1,299 lines long and contains:

1. Dependency scope definitions (DependencyScope enum)
2. Dependency provider implementation (DependencyProvider class)
3. Scope management (SessionScope and RequestScope classes)
4. Dependency injection utilities (DependencyInjector class and inject_dependencies decorator)
5. Helper functions for dependency management

## Refactoring Goals

1. **Improve Maintainability**: Split the large file into smaller, focused modules
2. **Enhance Organization**: Group related functionality into dedicated modules
3. **Improve Documentation**: Add comprehensive docstrings to all modules and classes
4. **Remove Backward Compatibility**: As specified in the requirements, no backward compatibility code will be included

## Refactoring Plan

### 1. Create Directory Structure

Create the following directory structure:
```
sifaka/core/dependency/
├── __init__.py
├── provider.py
├── scopes.py
├── injector.py
└── utils.py
```

### 2. Implement Modules

#### 2.1 `provider.py`

**Purpose**: Define the dependency provider class and related functionality.

**Content**:
- `DependencyProvider`: Singleton class for registering and retrieving dependencies
- Helper methods for dependency registration and retrieval
- Dependency graph management for circular dependency detection

#### 2.2 `scopes.py`

**Purpose**: Define dependency scopes and scope management classes.

**Content**:
- `DependencyScope`: Enum defining dependency lifecycles (singleton, session, request, transient)
- `SessionScope`: Context manager for session-scoped dependencies
- `RequestScope`: Context manager for request-scoped dependencies
- Scope-related utility functions

#### 2.3 `injector.py`

**Purpose**: Provide dependency injection utilities.

**Content**:
- `DependencyInjector`: Utility class for manual dependency injection
- `inject_dependencies`: Decorator for automatic dependency injection
- Injection-related utility functions

#### 2.4 `utils.py`

**Purpose**: Provide utility functions for dependency management.

**Content**:
- `provide_dependency`: Register a dependency
- `provide_factory`: Register a factory function
- `get_dependency`: Get a dependency by name
- `get_dependency_by_type`: Get a dependency by type
- `create_session_scope`: Create a session scope
- `create_request_scope`: Create a request scope
- `clear_dependencies`: Clear all dependencies

#### 2.5 `__init__.py`

**Purpose**: Export all public classes and functions.

**Content**:
- Import and export all public classes and functions from the other modules
- No backward compatibility code

### 3. Update Imports

Identify files that import from `sifaka.core.dependency` and update them to use the new module structure.

### 4. Create Tests

Create unit tests for each module to verify that all functionality is preserved.

## Implementation Strategy

1. **Incremental Implementation**: Implement one module at a time, starting with `provider.py`
2. **Comprehensive Testing**: Test each module thoroughly before moving to the next
3. **Documentation First**: Write comprehensive docstrings before implementing functionality
4. **No Backward Compatibility**: As specified in the requirements, no backward compatibility code will be included

## Success Metrics

1. **File Size Reduction**: Each module should be less than 300 lines
2. **Improved Organization**: Related functionality should be grouped together
3. **Enhanced Documentation**: All modules and classes should have comprehensive docstrings
4. **Test Coverage**: All modules should have at least 80% test coverage
5. **No Backward Compatibility**: No backward compatibility code should be included

## Timeline

1. **Create Directory Structure**: 1 hour
2. **Implement Provider Module**: 3 hours
3. **Implement Scopes Module**: 2 hours
4. **Implement Injector Module**: 2 hours
5. **Implement Utils Module**: 2 hours
6. **Implement Init Module**: 1 hour
7. **Update Imports**: 2 hours
8. **Create Tests**: 4 hours

**Total Estimated Time**: 17 hours

## Dependencies and Risks

### Dependencies
- Understanding of the current dependency injection system
- Knowledge of how components use the dependency injection system
- Awareness of any circular dependencies in the codebase

### Risks
- Breaking changes to the dependency injection API
- Circular dependencies between the new modules
- Impact on component initialization and lifecycle management

## Mitigation Strategies

1. **Thorough Testing**: Create comprehensive tests for all functionality
2. **Incremental Implementation**: Implement one module at a time
3. **Documentation**: Document all changes and update existing documentation
4. **Code Review**: Review all changes for potential issues
