# ADR-001: Single Function API Design

## Status
Accepted

## Context
When designing the public API for Sifaka, we needed to decide between multiple approaches:
1. A single `improve()` function that handles all use cases
2. Multiple specialized functions for different use cases
3. A class-based API with various methods

The primary use case is simple text improvement where users want to pass text and get back improved text with minimal complexity.

## Decision
We will implement a single `improve()` function as the primary public API.

```python
from sifaka import improve

# Simple usage
result = await improve("Your text here")

# With configuration
result = await improve(
    "Your text here",
    critics=["reflexion", "self_rag"],
    max_iterations=3,
    storage=storage_backend
)
```

## Rationale
1. **Simplicity**: A single function is easier to understand and remember
2. **Discoverability**: Users don't need to learn multiple function names
3. **Flexibility**: All configuration options are available through parameters
4. **Consistency**: Similar to other popular libraries (e.g., `requests.get()`)
5. **Gradual complexity**: Users can start simple and add complexity as needed

## Consequences

### Positive
- Very low barrier to entry for new users
- Consistent API across all use cases
- Easy to document and explain
- Follows the principle of "simple things should be simple"

### Negative
- The function signature might become complex with many optional parameters
- Advanced users might prefer more granular control
- IDE auto-completion might be overwhelming with many parameters

### Mitigation
- Use sensible defaults for all parameters
- Provide configuration objects for complex scenarios
- Document common patterns and examples
- Consider factory functions for common configurations

## Implementation Notes
- The function should be async to support LLM API calls
- Parameters should use type hints for better IDE support
- Return type should be a rich result object with metadata
- Error handling should be consistent and informative

## Related Decisions
- [ADR-002: Plugin Architecture](002-plugin-architecture.md)
- [ADR-003: Memory Management](003-memory-management.md)
