# Sifaka Component Simplification Plan

This document outlines a comprehensive plan for simplifying overly complicated components in the Sifaka codebase. The plan focuses on the chain, retrieval, and classifiers components, which have been identified as having excessive complexity.

## Prioritization

Components will be simplified in the following order:

1. **Chain Component** - Highest priority due to its central role in the framework
2. **Classifiers Component** - Medium-high priority, with many redundant implementations
3. **Retrieval Component** - Medium priority, with complex configuration and result models

## 1. Chain Component Simplification

### 1.1 Remove v2 References and Legacy Code

**Tasks:**
- Remove all references to "v2" in imports, docstrings, and examples
- Update documentation to reflect current architecture
- Ensure consistent naming across the codebase

**Implementation Approach:**
- Search for all "v2" references in the chain component
- Update import statements and docstrings
- Remove any legacy code that's no longer needed

### 1.2 Simplify Engine Class

**Tasks:**
- Split the Engine class into smaller, focused classes
- Move caching logic to a separate CacheManager class
- Move retry logic to a separate RetryManager class
- Simplify the run method to focus on core orchestration

**Implementation Approach:**
- Create new classes for specific responsibilities
- Refactor the Engine class to use these new classes
- Ensure backward compatibility with existing code
- Update tests to reflect the new structure

### 1.3 Streamline Interfaces

**Tasks:**
- Reduce the number of interfaces to essential ones
- Consolidate Model, Validator, Improver interfaces into simpler abstractions
- Remove redundant adapter classes

**Implementation Approach:**
- Identify core functionality needed in each interface
- Create simplified interfaces that focus on essential methods
- Update implementations to use the new interfaces
- Remove adapter classes that are no longer needed

### 1.4 Simplify State Management

**Tasks:**
- Reduce the complexity of the StateTracker class
- Simplify state update and retrieval methods
- Remove unnecessary history tracking if not used

**Implementation Approach:**
- Analyze current usage of state management
- Identify essential state management functionality
- Create a simplified StateManager class
- Update components to use the simplified state management

## 2. Classifiers Component Simplification

### 2.1 Flatten Implementation Structure

**Tasks:**
- Reorganize the implementation directory structure
- Move implementations to a flatter structure
- Remove redundant adapter classes
- Consolidate similar implementations

**Implementation Approach:**
- Create a new, flatter directory structure
- Move implementations to the new structure
- Update imports and references
- Remove redundant code

### 2.2 Simplify Classifier Interfaces

**Tasks:**
- Simplify the ClassifierImplementation interface
- Remove redundant async methods where not necessary
- Standardize method signatures across implementations

**Implementation Approach:**
- Create a simplified ClassifierInterface
- Update implementations to use the new interface
- Remove unnecessary methods and parameters
- Ensure consistent method signatures

### 2.3 Streamline Engine and Error Handling

**Tasks:**
- Simplify the Engine class to focus on core classification logic
- Standardize error handling across the component
- Reduce the number of custom error classes

**Implementation Approach:**
- Refactor the Engine class to focus on core functionality
- Create a standardized error handling approach
- Consolidate error classes into a simpler hierarchy
- Update implementations to use the standardized approach

## 3. Retrieval Component Simplification

### 3.1 Simplify Configuration

**Tasks:**
- Consolidate multiple configuration classes into a simpler structure
- Use sensible defaults to reduce configuration complexity
- Remove unnecessary configuration options

**Implementation Approach:**
- Analyze current configuration usage
- Create a simplified configuration structure
- Provide sensible defaults for common use cases
- Update components to use the simplified configuration

### 3.2 Streamline Result Models

**Tasks:**
- Simplify the result model hierarchy
- Reduce the use of generic typing where not necessary
- Create simpler, more focused result classes

**Implementation Approach:**
- Analyze current result model usage
- Create simplified result models
- Update components to use the simplified models
- Ensure backward compatibility with existing code

### 3.3 Reduce Directory Nesting

**Tasks:**
- Flatten the directory structure to reduce complexity
- Move implementations to a simpler structure
- Update imports and references

**Implementation Approach:**
- Create a new, flatter directory structure
- Move implementations to the new structure
- Update imports and references
- Remove redundant code

## Implementation Strategy

For each component, we will follow this implementation strategy:

1. **Analysis Phase**
   - Review current usage patterns
   - Identify essential functionality
   - Document dependencies and interfaces

2. **Design Phase**
   - Create simplified design
   - Document new interfaces and classes
   - Define migration path

3. **Implementation Phase**
   - Create new simplified classes and interfaces
   - Update existing code to use new components
   - Remove redundant code
   - Update documentation

4. **Testing Phase**
   - Create tests for new components
   - Verify existing functionality works
   - Check for regressions

5. **Documentation Phase**
   - Update documentation to reflect new design
   - Provide migration examples
   - Document rationale for changes

## Testing Strategy

For each simplification task, we will:

1. Create unit tests for new components
2. Ensure existing tests pass with the new implementation
3. Create integration tests to verify component interactions
4. Test with real-world examples from the examples directory

## Success Criteria

The simplification will be considered successful if:

1. Code complexity is reduced (fewer lines, simpler methods)
2. Component interactions are more straightforward
3. Documentation is clearer and more concise
4. All tests pass
5. Existing functionality is preserved

## Timeline

1. **Chain Component Simplification**: 1-2 days
2. **Classifiers Component Simplification**: 1-2 days
3. **Retrieval Component Simplification**: 1-2 days

Total estimated time: 3-6 days

## Progress Tracking

We will track progress in the SUMMARY.md file, adding a new section for component simplification.
