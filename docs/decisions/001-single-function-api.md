# ADR-001: Single Function API

## Status
Accepted

## Context
The original Sifaka codebase had multiple complex APIs (`SifakaValidator`, `SifakaImprover`, `SifakaAnalyzer`) that created confusion and cognitive overhead for users. User feedback indicated that the API was too complex and difficult to understand.

## Decision
We will provide a single `improve()` function as the primary API for all text improvement operations.

```python
async def improve(
    text: str,
    *,
    critics: Optional[List[str]] = None,
    max_iterations: int = 3,
    validators: Optional[List[Validator]] = None,
    config: Optional[Config] = None,
    storage: Optional[StorageBackend] = None,
) -> SifakaResult
```

## Rationale

### Simplicity
- **Single entry point** eliminates confusion about which API to use
- **Keyword-only arguments** prevent positional argument mistakes
- **Sensible defaults** allow immediate usage without configuration
- **One import** (`from sifaka import improve`) is all users need

### Discoverability
- **IDE autocomplete** shows all available options in one place
- **Documentation** can focus on a single, well-defined interface
- **Examples** are consistent and transferable across use cases

### Flexibility
- **Full configurability** through parameters supports advanced use cases
- **Optional parameters** allow progressive complexity
- **Type hints** provide clear contracts and IDE support

## Consequences

### Positive
- **Reduced cognitive load** for new users
- **Faster onboarding** with immediate value
- **Consistent usage patterns** across all applications
- **Better documentation** focused on single interface
- **Easier testing** with fewer API surfaces

### Negative
- **Parameter growth** as features are added (mitigated by keyword-only arguments)
- **Less granular control** compared to class-based APIs (mitigated by comprehensive parameters)

## Alternatives Considered

### Multiple Specialized Functions
```python
await improve_with_reflexion(text)
await improve_with_constitutional(text)
await improve_with_multiple_critics(text, critics=[...])
```

**Rejected because**: Creates API proliferation and versioning complexity.

### Class-Based Builder Pattern
```python
improver = SifakaImprover().with_critics(["reflexion"]).with_max_cost(1.0)
result = await improver.improve(text)
```

**Rejected because**: Adds complexity without clear benefits for the common case.

### Configuration Object Pattern
```python
config = SifakaConfig(critics=["reflexion"], max_cost=1.0)
result = await improve(text, config=config)
```

**Rejected because**: Adds indirection for simple use cases, though this pattern is used internally.

## Implementation Notes

### Backward Compatibility
This is a breaking change from previous versions. Migration is straightforward:

**Before**:
```python
validator = SifakaValidator(...)
improver = SifakaImprover(...)
result = improver.improve(text)
```

**After**:
```python
result = await improve(text, critics=["reflexion"], max_iterations=3)
```

### Internal Architecture
The `improve()` function internally creates a `SifakaEngine` with appropriate configuration, maintaining clean separation of concerns while presenting a simple interface.

### Future Extensions
New capabilities can be added as parameters to the `improve()` function without breaking existing code, thanks to keyword-only arguments and optional parameters.

## References
- User feedback indicating API complexity issues
- [The Zen of Python](https://www.python.org/dev/peps/pep-0020/): "Simple is better than complex"
- [PEP 3102](https://www.python.org/dev/peps/pep-3102/): Keyword-only arguments