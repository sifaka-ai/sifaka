# File Structure Refactoring Plan

## Overview

Several files in the Sifaka codebase exceed 1,000 lines, making them harder to maintain. This document outlines a plan to split these large files into smaller, more focused modules while preserving functionality and interfaces.

## Problem Statement

Large files present several challenges:

1. **Cognitive Load**: Understanding a 2,000+ line file requires significant mental effort
2. **Navigation Difficulty**: Finding specific functionality in large files is challenging
3. **Testing Complexity**: Large files with multiple responsibilities are harder to test thoroughly
4. **Collaboration Challenges**: Multiple developers working on the same large file can lead to merge conflicts
5. **Maintenance Burden**: Updates to large files require understanding more context

## Target Files

The following files have been identified as candidates for refactoring based on their size:

| File Path | Line Count | Complexity | Priority |
|-----------|------------|------------|----------|
| sifaka/utils/config.py | 2,810 | High | High |
| sifaka/critics/implementations/lac.py | 1,938 | Medium | Medium |
| sifaka/rules/formatting/format.py | 1,733 | Medium | Medium |
| sifaka/rules/formatting/style.py | 1,625 | Medium | Medium |
| sifaka/utils/errors.py | 1,444 | High | High |
| sifaka/critics/implementations/prompt.py | 1,400 | Medium | Medium |
| sifaka/critics/base.py | 1,306 | High | High |
| sifaka/core/dependency.py | 1,299 | High | High |
| sifaka/models/base.py | 1,185 | High | Medium |
| sifaka/adapters/classifier/adapter.py | 1,176 | Medium | Low |

## Detailed Refactoring Plans

### 1. sifaka/utils/config.py (2,810 lines)

**Current Structure**: Single file containing all configuration classes and standardization functions.

**Proposed Structure**:
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

**Key Considerations**:
- Update all imports throughout the codebase
- Maintain backward compatibility through __init__.py exports
- Ensure no circular dependencies are created
- Comprehensive testing to verify functionality

**Estimated Effort**: High (3-5 days)

### 2. sifaka/critics/implementations/lac.py (1,938 lines)

**Current Structure**: Single file containing FeedbackCritic, ValueCritic, and LACCritic implementations.

**Proposed Structure**:
```
sifaka/critics/implementations/lac/
├── __init__.py         # Exports and factory functions
├── feedback.py         # FeedbackCritic implementation
├── value.py            # ValueCritic implementation
├── combined.py         # LACCritic implementation
└── prompts.py          # Default prompts and templates
```

**Key Considerations**:
- Ensure consistent state management across split files
- Update imports in files that use LAC critics
- Maintain factory function interfaces

**Estimated Effort**: Medium (2-3 days)

### 3. sifaka/rules/formatting/format.py (1,733 lines)

**Current Structure**: Single file containing format validation rules, validators, and configurations.

**Proposed Structure**:
```
sifaka/rules/formatting/format/
├── __init__.py         # Exports and factory functions
├── config.py           # Format configuration classes
├── validators.py       # Format validators
├── rules.py            # Format rule implementations
└── json.py             # JSON-specific formatting rules
```

**Key Considerations**:
- Manage dependencies between rule and validator classes
- Update imports in files that use format rules
- Maintain factory function interfaces

**Estimated Effort**: Medium (2-3 days)

### 4. sifaka/rules/formatting/style.py (1,625 lines)

**Current Structure**: Single file containing style validation rules, validators, and configurations.

**Proposed Structure**:
```
sifaka/rules/formatting/style/
├── __init__.py         # Exports and factory functions
├── config.py           # Style configuration classes
├── validators.py       # Style validators
├── rules.py            # Style rule implementations
└── analyzers.py        # Style analysis utilities
```

**Key Considerations**:
- Similar to format.py refactoring
- Ensure consistent interfaces with format rules

**Estimated Effort**: Medium (2-3 days)

### 5. sifaka/utils/errors.py (1,444 lines)

**Current Structure**: Single file containing all error classes.

**Proposed Structure**:
```
sifaka/utils/errors/
├── __init__.py         # Exports and base exceptions
├── validation.py       # Validation-related errors
├── configuration.py    # Configuration-related errors
├── runtime.py          # Runtime errors
├── dependency.py       # Dependency-related errors
└── io.py               # I/O-related errors
```

**Key Considerations**:
- Widespread usage throughout codebase
- Maintain proper exception hierarchy
- Comprehensive testing of error handling

**Estimated Effort**: High (3-4 days)

### 6. sifaka/critics/implementations/prompt.py (1,400 lines)

**Current Structure**: Single file containing prompt critic implementations.

**Proposed Structure**:
```
sifaka/critics/implementations/prompt/
├── __init__.py         # Exports and factory functions
├── base.py             # Base prompt critic implementation
├── templates.py        # Prompt templates
├── validators.py       # Prompt validation utilities
└── processors.py       # Response processing utilities
```

**Key Considerations**:
- Similar to lac.py refactoring
- Ensure consistent interfaces with other critics

**Estimated Effort**: Medium (2-3 days)

### 7. sifaka/critics/base.py (1,306 lines)

**Current Structure**: Single file containing base critic classes and interfaces.

**Proposed Structure**:
```
sifaka/critics/base/
├── __init__.py         # Exports and base classes
├── interfaces.py       # Critic interfaces
├── validators.py       # Base validator implementations
├── improvers.py        # Base improver implementations
└── factories.py        # Factory functions
```

**Key Considerations**:
- Core component with many dependencies
- Maintain consistent interfaces
- Comprehensive testing required

**Estimated Effort**: High (3-4 days)

### 8. sifaka/core/dependency.py (1,299 lines)

**Current Structure**: Single file containing dependency injection system.

**Proposed Structure**:
```
sifaka/core/dependency/
├── __init__.py         # Exports and DependencyProvider
├── scopes.py           # Dependency scopes
├── decorators.py       # Dependency injection decorators
├── registry.py         # Dependency registry implementation
└── graph.py            # Dependency graph management
```

**Key Considerations**:
- Core infrastructure used throughout codebase
- High risk of creating circular imports
- Comprehensive testing required

**Estimated Effort**: High (3-5 days)

## Implementation Strategy

### Phase 1: Planning and Preparation
1. **Create Detailed Specifications**: For each file, create detailed specifications for the split
2. **Identify Import Dependencies**: Map all imports to understand impact
3. **Create Test Plan**: Develop comprehensive test plan for each refactoring
4. **Set Up CI/CD**: Ensure CI/CD pipeline can validate refactoring

### Phase 2: Implementation
1. **Start with Lower-Risk Files**: Begin with files that have fewer imports elsewhere
2. **One File at a Time**: Complete the refactoring of one file before moving to the next
3. **Follow This Process**:
   - Create new directory structure
   - Move code to appropriate files
   - Update imports in the new files
   - Update __init__.py to export all public interfaces
   - Update imports in dependent files
   - Run comprehensive tests
   - Address any issues before proceeding

### Phase 3: Validation and Documentation
1. **Comprehensive Testing**: Run full test suite after each file refactoring
2. **Documentation Updates**: Update documentation to reflect new structure
3. **Code Review**: Conduct thorough code review of changes
4. **Performance Testing**: Verify no performance regressions

## Risk Mitigation

1. **Backward Compatibility**: Maintain all public interfaces
2. **Incremental Changes**: Implement changes one file at a time
3. **Comprehensive Testing**: Ensure thorough test coverage
4. **Rollback Plan**: Have a clear rollback strategy for each change
5. **Documentation**: Update documentation to reflect new structure

## Timeline and Resources

**Estimated Total Effort**: 20-30 developer days

**Suggested Prioritization**:
1. Start with utils/config.py (highest impact)
2. Move to utils/errors.py (widespread usage)
3. Continue with core/dependency.py (core infrastructure)
4. Address remaining files based on development priorities

## Conclusion

This refactoring will significantly improve the maintainability of the Sifaka codebase by breaking down large files into smaller, more focused modules. While the effort required is substantial, the long-term benefits in terms of maintainability, testability, and developer productivity will justify the investment.

The refactoring should be approached methodically, with careful planning, incremental implementation, and comprehensive testing to minimize risks and ensure a smooth transition.
