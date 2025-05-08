# State Management Examples

This directory contains examples demonstrating the standardized state management patterns in Sifaka.

## Standardized Initialization Pattern

The `standardized_initialization.py` example demonstrates the standard initialization pattern for Sifaka components, showing both basic initialization in `__init__()` and resource-intensive initialization in `warm_up()`.

### Key Concepts

1. **Two-Phase Initialization**:
   - Basic initialization in `__init__()` for lightweight setup
   - Resource-intensive initialization in `warm_up()` for heavy resources

2. **State Management**:
   - Using `_state_manager` with `StateManager` for consistent state handling
   - Proper initialization and access patterns

3. **Error Handling**:
   - Proper error handling during initialization
   - Graceful recovery from initialization failures

### Running the Example

```bash
python standardized_initialization.py
```

### Expected Output

```
Basic initialization complete for example_component
Component name: example_component
Component description: An example component with standardized initialization
Warming up example_component...
Loading model for example_component...
Loading embeddings for example_component...
Component example_component successfully warmed up
Processing Hello, world! with example_component
Result: Processed by example_component: Hello, world!
Using cached result for Hello, world!
Result: Processed by example_component: Hello, world!
```

## Key Takeaways

1. **Consistent Pattern**: All components should follow the same initialization pattern
2. **Separation of Concerns**: Separate lightweight and heavyweight initialization
3. **Error Handling**: Always handle initialization errors properly
4. **State Access**: Always access state through `_state_manager.get_state()`
5. **Caching**: Use state for caching to improve performance

## Related Documentation

- [State Management in Sifaka](/docs/state_management.md)
- [State Management Standardization Plan](/docs/implementation_notes/state_management_standardization_plan.md)
