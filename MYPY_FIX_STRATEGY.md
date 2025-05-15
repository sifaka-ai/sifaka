# MYPY Error Fix Strategy

## Analysis of Current Errors

After running `mypy . --show-error-codes`, I found approximately 646 errors across 85 files in the codebase. These errors can be categorized into the following types:

1. **Type Annotation Issues (high frequency)**
   - Missing type annotations for variables, function parameters, and return types
   - Incompatible return types (returning `Any` when a specific type is declared)
   - Improperly typed function arguments

2. **Inheritance and Interface Issues**
   - Incompatible method signatures in subclasses
   - Abstract class instantiation issues
   - Liskov substitution principle violations

3. **Attribute Access Errors**
   - Accessing attributes that don't exist or are misspelled
   - Union type attribute access issues

4. **Operational Errors**
   - Unsupported operand types (especially with float/str combinations)
   - Unreachable code statements

5. **Configuration and Initialization Errors**
   - Unexpected keyword arguments
   - Missing required positional arguments
   - Improper `dict` usage where typed objects are expected

## Prioritization Strategy

To efficiently address these errors, I'll approach them in the following order:

1. **Foundation Fixes First**
   - Fix import errors and module structure issues
   - Address abstract class implementations
   - Fix base class interfaces before derived classes

2. **High-Impact, Low-Effort Fixes**
   - Add missing type annotations to variables and function signatures
   - Fix type declaration conflicts
   - Convert implicit Optional types to explicit ones

3. **Common Pattern Errors**
   - Address repetitive error patterns (like operator type issues)
   - Fix consistent attribute access errors

4. **Complex Type System Issues**
   - Handle inheritance hierarchy issues
   - Fix protocol implementations
   - Address TypedDict and generic type issues

5. **Gradual Migration Path**
   - Use strategic `# type: ignore` comments for temporary relief
   - Begin with most used/critical modules first
   - Create helper types for common patterns

## Implementation Plan

### Phase 1: Setup and Structure (Day 1)
- Create proper type stubs for any missing dependencies
- Fix imports and module structure issues
- Address "unreachable code" errors
- Establish base class interface consistency

### Phase 2: Core Type Annotations (Days 2-3)
- Add missing type annotations to variables
- Fix return type declarations
- Address incompatible argument type errors
- Create helper utility types for common patterns

### Phase 3: Fix Operational and Attribute Errors (Days 4-5)
- Fix operator type mismatches
- Correct attribute access errors
- Ensure proper typing for class attributes and methods
- Fix union type handling issues

### Phase 4: Configuration and Advanced Types (Days 6-7)
- Fix configuration class mismatches
- Ensure proper TypedDict implementations
- Address generic type parameter issues
- Fix dictionary vs. object typing issues

### Phase 5: Testing and Validation (Days 8-9)
- Create tests to validate fixes don't break functionality
- Set up CI/CD to prevent regression
- Document type practices for the codebase
- Address any remaining issues

## Tools and Techniques

1. **Automated Refactoring Tools**
   - Use `pytype` to infer types where mypy is struggling
   - Consider `monkeytype` for runtime type collection
   - Use IDEs with good Python type checking support

2. **Documentation Improvements**
   - Document type usage patterns as they're fixed
   - Create examples of proper type usage for team reference

3. **CI Integration**
   - Add mypy to CI pipeline with graduated strictness
   - Create pre-commit hooks for type checking

4. **Custom Type Utils**
   - Create helper types for recurring patterns
   - Use TypeVar and Protocol for complex cases

## Specific Modules to Focus On

Based on error frequency, these modules need the most attention:
- `sifaka/critics/implementations/*` (particularly `lac.py`, `prompt.py`)
- `sifaka/rules/formatting/format/*`
- `sifaka/rules/content/base.py`
- `sifaka/classifiers/*`
- `sifaka/adapters/*`
- `sifaka/retrieval/implementations/simple.py`

## Best Practices to Enforce

1. Always annotate function parameters and return types
2. Use explicit Optional[] rather than allowing None as default
3. Define proper protocols for duck-typed interfaces
4. Create typed configurations with proper inheritance
5. Use consistent patterns for error handling with proper typing
6. Avoid type: Any except where absolutely necessary

## Monitoring and Measurement

- Track reduction in error count over time
- Monitor impact on code quality metrics
- Ensure test coverage for fixed modules
- Check for runtime errors that might indicate type issues