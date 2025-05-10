# Common Utilities Examples

This directory contains examples demonstrating how to use the common utilities in Sifaka components.

## Overview

The examples show the standardized implementation patterns for Sifaka components, including:

- State management
- Error handling
- Result creation
- Caching
- Statistics tracking

## Examples

### Component Example

The `component_example.py` file demonstrates a complete component implementation using the common utilities:

```python
from sifaka.utils.state import StateManager
from sifaka.utils.common import (
    initialize_component_state,
    get_cached_result,
    update_cache,
    update_statistics,
    safely_execute,
    create_standard_result
)

class ExampleComponent(BaseModel):
    # Component implementation using common utilities
    # ...
```

Key features demonstrated:

1. **State Initialization**: Using `initialize_component_state` to set up component state
2. **Caching**: Using `get_cached_result` and `update_cache` for result caching
3. **Error Handling**: Using `safely_execute` for standardized error handling
4. **Statistics Tracking**: Using `update_statistics` to track execution statistics
5. **Result Creation**: Using `create_standard_result` for standardized results

## Running the Examples

To run the examples:

```bash
# Run the component example
python -m examples.common_utilities.component_example
```

## Best Practices

When implementing your own components:

1. **Follow the standardized pattern**: Use the pattern demonstrated in these examples
2. **Use the common utilities**: Don't reinvent the wheel - use the provided utilities
3. **Separate configuration from state**: Use Pydantic models for configuration
4. **Handle errors consistently**: Use the error handling utilities
5. **Document component behavior**: Document how your component uses state, handles errors, etc.
