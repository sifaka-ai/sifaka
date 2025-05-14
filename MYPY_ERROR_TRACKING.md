# Mypy Error Tracking

This document tracks the progress of fixing mypy type checking errors in the Sifaka codebase.

## Current Status

- **Total files with errors**: 69
- **Files fixed**: 43
- **Remaining files**: 69
- **Current error count**: 1,019

## Error Categories and Solutions

### 1. Type Variable Variance Issues

**Description**: Issues with type variable variance in protocols and generic classes.

**Common errors**:
- Invariant type variables used in protocols where covariant ones are expected
- Invariant type variables used in protocols where contravariant ones are expected
- Contravariant type variables used in protocols where invariant ones are expected
- Covariant type variables used in protocols where invariant ones are expected

**Solution patterns**:
- Use `Protocol[+T]` for covariant type variables (output types)
- Use `Protocol[-T]` for contravariant type variables (input types)
- Ensure all type variables are properly listed in `Generic[...]` or `Protocol[...]`
- Update protocol definitions to use proper variance annotations

### 2. Missing Type Annotations

**Description**: Functions and variables missing proper type annotations.

**Common errors**:
- Functions missing type annotations for one or more arguments
- Variables missing type annotations
- Missing return type annotations

**Solution patterns**:
- Add proper type annotations to function parameters
- Add return type annotations where missing
- Use appropriate type hints from `typing` module
- Use `Any` only when absolutely necessary

### 3. Return Type Incompatibilities

**Description**: Functions returning types that don't match their declared return types.

**Common errors**:
- Incompatible return value types
- Returning `Any` from functions with specific return types
- Returning `None` from functions with non-optional return types

**Solution patterns**:
- Fix return types to match declared types
- Update function implementations to ensure correct return types
- Use Union types in return annotations where necessary
- Consider refactoring functions with inconsistent return types

### 4. Attribute Errors

**Description**: Code accessing attributes that don't exist on objects.

**Common errors**:
- Accessing attributes that don't exist
- Accessing attributes on potentially None values
- Accessing attributes on union types without type guards

**Solution patterns**:
- Fix attribute access patterns
- Update code to use correct attribute names
- Add proper type guards for union types
- Use `isinstance()` checks before accessing attributes on union types

### 5. Assignment Errors

**Description**: Assigning values of incompatible types.

**Common errors**:
- Incompatible types in assignments
- Assigning `None` to non-optional variables
- Assigning complex types to simpler types

**Solution patterns**:
- Fix type mismatches in assignments
- Add explicit type conversions where needed
- Update variable type annotations to match assigned values
- Refactor code to avoid type mismatches

### 6. Argument Type Errors

**Description**: Arguments passed to functions with incompatible types.

**Common errors**:
- Incompatible argument types
- Missing named arguments
- Unexpected keyword arguments

**Solution patterns**:
- Fix argument types to match function signatures
- Add type conversions where needed
- Update function calls to provide correct argument types
- Consider refactoring functions with complex argument requirements

### 7. Method Override Incompatibilities

**Description**: Method signatures incompatible with their supertype.

**Common errors**:
- Parameter types incompatible with supertype
- Return types incompatible with supertype
- Missing parameters in overridden methods

**Solution patterns**:
- Ensure method signatures match their supertype
- Use covariant return types where appropriate
- Use contravariant parameter types where appropriate

## Implementation Plan

### Phase 1: Core Type Definitions (High Priority)

**Focus**: Fix type variable variance issues in protocol definitions

**Files to address**:
- [x] sifaka/interfaces/chain/managers/validation.py
- [x] sifaka/interfaces/chain/managers/formatter.py
- [x] sifaka/interfaces/rule.py
- [x] sifaka/interfaces/critic.py
- [x] sifaka/interfaces/classifier.py
- [x] sifaka/critics/base/protocols.py
- [x] sifaka/core/base.py

**Expected outcome**: Resolve approximately 20-30% of type errors

### Phase 2: Function Signatures (High Priority)

**Focus**: Fix missing type annotations and incompatible return types

**Files to address**:
- [x] sifaka/interfaces/model.py
- [x] sifaka/utils/logging.py
- [x] sifaka/core/dependency/scopes.py
- [x] sifaka/core/dependency/injector.py
- [x] sifaka/classifiers/implementations/adapters.py
- [x] sifaka/rules/content/safety.py
- [x] sifaka/models/providers/gemini.py

**Expected outcome**: Resolve approximately 30-40% of type errors

### Phase 3: Type Usage (Medium Priority)

**Focus**: Fix attribute errors, assignment errors, and argument type errors

**Files to address**:
- [x] sifaka/utils/base_results.py
- [x] sifaka/utils/text.py
- [x] sifaka/utils/tracing.py
- [x] sifaka/core/initialization.py
- [ ] sifaka/chain/interfaces.py
- [ ] sifaka/utils/errors/safe_execution.py
- [x] sifaka/core/results.py
- [ ] sifaka/chain/state.py
- [ ] sifaka/utils/results.py
- [ ] sifaka/models/managers/client.py
- [ ] sifaka/critics/core.py

**Expected outcome**: Resolve approximately 20-30% of type errors

### Phase 4: Final Cleanup (Medium Priority)

**Focus**: Fix remaining errors and ensure consistency

**Files to address**:
- [ ] sifaka/critics/base/metadata.py
- [ ] sifaka/models/utils.py
- [ ] sifaka/utils/results.py
- [ ] sifaka/core/base.py
- [ ] sifaka/adapters/chain/formatter.py
- [ ] sifaka/rules/content/safety.py

**Expected outcome**: Resolve all remaining type errors

## Progress Tracking

### Interfaces (8/8 completed)

- [x] sifaka/interfaces/chain/managers/validation.py
- [x] sifaka/interfaces/chain/managers/formatter.py
- [x] sifaka/interfaces/rule.py
- [x] sifaka/interfaces/model.py
- [x] sifaka/interfaces/critic.py
- [x] sifaka/interfaces/retrieval.py
- [x] sifaka/interfaces/chain/chain.py
- [x] sifaka/interfaces/classifier.py

### Core (6/6 completed)

- [x] sifaka/core/base.py (High Priority)
- [x] sifaka/core/results.py (Medium Priority)
- [x] sifaka/core/managers/memory.py (High Priority)
- [x] sifaka/core/dependency/scopes.py (Medium Priority)
- [x] sifaka/core/dependency/provider.py (Medium Priority)
- [x] sifaka/core/dependency/injector.py (Medium Priority)

### Utils (21/21 completed)

- [x] sifaka/utils/logging.py (High Priority)
- [x] sifaka/utils/base_results.py (Medium Priority)
- [x] sifaka/utils/tracing.py (Medium Priority)
- [x] sifaka/utils/state.py (High Priority)
- [x] sifaka/utils/config/retrieval.py (Low Priority)
- [x] sifaka/utils/config/classifiers.py (Low Priority)
- [x] sifaka/utils/config/chain.py (Low Priority)
- [x] sifaka/utils/config/rules.py (Low Priority)
- [x] sifaka/utils/config/models.py (Low Priority)
- [x] sifaka/utils/config/critics.py (High Priority)
- [x] sifaka/utils/errors/safe_execution.py (Medium Priority)
- [x] sifaka/utils/errors/results.py (Medium Priority)
- [x] sifaka/utils/results.py (Medium Priority)
- [x] sifaka/utils/text.py (Medium Priority)
- [x] sifaka/utils/config/base.py (Low Priority)
- [x] sifaka/utils/config/factories.py (Low Priority)
- [x] sifaka/utils/config/formatters.py (Low Priority)
- [x] sifaka/utils/config/improvers.py (Low Priority)
- [x] sifaka/utils/config/providers.py (Low Priority)
- [x] sifaka/utils/config/state.py (Low Priority)
- [x] sifaka/utils/config/validators.py (Low Priority)

### Models (0/9 completed)

- [ ] sifaka/models/utils.py (Medium Priority)
- [ ] sifaka/models/managers/tracing.py (Low Priority)
- [ ] sifaka/models/managers/client.py (High Priority)
- [ ] sifaka/models/managers/openai_token_counter.py (Low Priority)
- [ ] sifaka/models/services/generation.py (Medium Priority)
- [ ] sifaka/models/base/factory.py (Medium Priority)
- [ ] sifaka/models/base/provider.py (High Priority)
- [ ] sifaka/models/providers/gemini.py (Low Priority)
- [ ] sifaka/models/factories.py (Medium Priority)

### Critics (1/6 completed)

- [x] sifaka/critics/base/metadata.py
- [ ] sifaka/critics/base/protocols.py (High Priority)
- [ ] sifaka/critics/core.py (High Priority)
- [ ] sifaka/critics/implementations/reflexion.py (Medium Priority)
- [ ] sifaka/critics/implementations/self_rag.py (Medium Priority)
- [ ] sifaka/critics/implementations/lac.py (Medium Priority)

### Chain (0/7 completed)

- [ ] sifaka/interfaces/chain/components/validator.py (Medium Priority)
- [ ] sifaka/interfaces/chain/components/improver.py (Medium Priority)
- [ ] sifaka/interfaces/chain/components/formatter.py (Medium Priority)
- [ ] sifaka/adapters/chain/formatter.py (Medium Priority)
- [ ] sifaka/adapters/chain/improver.py (Medium Priority)
- [ ] sifaka/adapters/chain/validator.py (Medium Priority)
- [ ] sifaka/adapters/chain/model.py (Medium Priority)

### Classifiers (0/2 completed)

- [ ] sifaka/classifiers/interfaces.py (Medium Priority)
- [ ] sifaka/classifiers/adapters.py (Medium Priority)

### Retrieval (0/6 completed)

- [ ] sifaka/retrieval/core.py (Medium Priority)
- [ ] sifaka/retrieval/result.py (Medium Priority)
- [ ] sifaka/retrieval/strategies/ranking.py (Low Priority)
- [ ] sifaka/retrieval/managers/query.py (Medium Priority)
- [ ] sifaka/retrieval/factories.py (Medium Priority)
- [ ] sifaka/retrieval/__init__.py (Low Priority)

### Rules (0/5 completed)

- [ ] sifaka/rules/utils.py (Medium Priority)
- [ ] sifaka/rules/validators.py (Medium Priority)
- [ ] sifaka/rules/content/safety.py (Medium Priority)
- [ ] sifaka/rules/managers/validation.py (Medium Priority)
- [ ] sifaka/rules/factories.py (Low Priority)

## Progress Summary

- **Total files with errors**: 112
- **Files fixed**: 32 (28.6%)
- **Remaining files**: 80 (71.4%)
- **High priority files fixed**: 11/25 (44%)
- **Medium priority files fixed**: 19/59 (32.2%)
- **Low priority files fixed**: 2/16 (12.5%)

## Common Error Patterns and Solutions

### Pattern 1: Conditional Attribute Access

**Problem**:
```python
if self and self.method():  # Error: Redundant condition
    ...
```

**Solution**:
```python
if self.method():  # Fixed: Direct method call
    ...
```

### Pattern 2: Optional Attribute Access

**Problem**:
```python
if self._state_manager and self._state_manager.get(...):  # Error: Redundant condition
    ...
```

**Solution**:
```python
if self._state_manager is not None and self._state_manager.get(...):  # Fixed: Explicit None check
    ...
```

### Pattern 3: Union Type Access

**Problem**:
```python
result = get_result()  # Returns Union[Dict[str, Any], str]
value = result.get("key")  # Error: Item "str" has no attribute "get"
```

**Solution**:
```python
result = get_result()
if isinstance(result, dict):  # Fixed: Type guard
    value = result.get("key")
```

## Next Steps

1. Focus on high-priority files first, especially those in core modules
2. Address common error patterns across multiple files
3. Fix type variable variance issues in critic protocols
4. Fix protocol type parameter issues in remaining interfaces
5. Fix return type incompatibilities in core modules
6. Update progress in this document after each file is fixed

## Files Fixed

1. `sifaka/core/dependency/provider.py` - Fixed missing type annotation for `_initialized` class variable
2. `sifaka/core/results.py` - Fixed double Optional types in function signatures (e.g., `Optional[Optional[List[str]]]` → `Optional[List[str]]`)
3. `sifaka/utils/config/critics.py` - Fixed incorrect type annotations
4. `sifaka/utils/config/models.py` - Fixed incorrect type annotations
5. `sifaka/utils/config/chain.py` - Fixed incorrect type annotations
6. `sifaka/utils/config/classifiers.py` - Fixed incorrect type annotations
7. `sifaka/utils/config/retrieval.py` - Fixed incorrect type annotations
8. `sifaka/utils/config/rules.py` - Fixed incorrect type annotations
9. `sifaka/utils/config/validators.py` - Fixed incorrect type annotations
10. `sifaka/utils/config/base.py` - Fixed incorrect type annotations
11. `sifaka/utils/config/factories.py` - Fixed incorrect type annotations
12. `sifaka/utils/config/formatters.py` - Fixed incorrect type annotations
13. `sifaka/utils/config/improvers.py` - Fixed incorrect type annotations
14. `sifaka/utils/config/providers.py` - Fixed incorrect type annotations
15. `sifaka/utils/config/state.py` - Fixed incorrect type annotations
16. `sifaka/core/results.py` - Fixed all instances of double Optional types in function parameters
17. `sifaka/utils/state.py` - Fixed type variable usage by adding bound=BaseModel and using Type from typing module
18. `sifaka/utils/text.py` - Fixed return type in is_empty_text function and converted int to str in metadata assignments
19. `sifaka/utils/base_results.py` - Fixed incompatible return value types in with_* methods and removed double Optional types in function parameters
20. `sifaka/utils/tracing.py` - Fixed dictionary usage where Pydantic models should be used and removed double Optional types
21. `sifaka/utils/errors/safe_execution.py` - Fixed return type annotations, removed double Optional types, and fixed unused imports
22. `sifaka/utils/errors/results.py` - Fixed return type annotations and improved function signature formatting
23. `sifaka/core/initialization.py` - Fixed type variable usage in generic methods, added CleanupError class, fixed return type issues, and improved type safety in class methods
24. `sifaka/utils/results.py` - Fixed missing required arguments for result classes, removed double Optional types, and fixed type incompatibilities in function parameters and return values
25. `sifaka/utils/resources.py` - Fixed undefined variable issues, incompatible type in assignment, and contextmanager return type
26. `sifaka/core/protocol.py` - Fixed missing type annotations for variables and return type issues in helper functions
27. `sifaka/core/factories.py` - Fixed DependencyProvider.get_by_type method errors by using get_dependency_by_type function, fixed classifier factory imports, and fixed type mismatches in create_rule and create_adapter functions
28. `sifaka/interfaces/classifier.py` - Fixed Protocol class definitions by removing variance annotations and adding proper return statements to Protocol method stubs
