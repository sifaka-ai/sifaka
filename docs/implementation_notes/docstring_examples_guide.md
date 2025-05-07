# Docstring Examples Guide

This document provides guidelines for creating comprehensive examples for Sifaka component docstrings. Good examples are crucial for helping users understand how to use components effectively.

## Principles for Effective Examples

1. **Start Simple**: Begin with the simplest possible example that demonstrates the component's core functionality.
2. **Build Complexity**: Add more complex examples that show advanced features or configurations.
3. **Show Integration**: Demonstrate how the component integrates with other Sifaka components.
4. **Include Error Handling**: Show how to handle common errors or edge cases.
5. **Use Real-World Scenarios**: When possible, use examples that reflect real-world use cases.
6. **Be Concise**: Keep examples focused and avoid unnecessary complexity.
7. **Ensure Correctness**: All examples should be runnable and produce the expected results.

## Example Template

```python
"""
Brief description of the component.

Detailed description of the component's purpose and functionality.

## Architecture
Description of the component's architecture and design patterns.

## Lifecycle
Description of the component's lifecycle (initialization, operation, cleanup).

## Error Handling
Description of how the component handles errors and exceptions.

## Examples

### Basic Usage
```python
# Import the component
from sifaka.component_type import create_component

# Create the component with basic configuration
component = create_component(param1="value1", param2="value2")

# Use the component
result = component.method(input_data)
print(f"Result: {result}")
```

### Advanced Configuration
```python
# Import the component and configuration
from sifaka.component_type import create_component, ComponentConfig

# Create a custom configuration
config = ComponentConfig(
    param1="value1",
    param2="value2",
    advanced_param="advanced_value"
)

# Create the component with custom configuration
component = create_component(config=config)

# Use the component with advanced features
result = component.advanced_method(input_data, extra_param="value")
print(f"Advanced result: {result}")
```

### Integration with Other Components
```python
# Import components
from sifaka.component_type import create_component
from sifaka.other_component import create_other_component

# Create components
component1 = create_component(param1="value1")
component2 = create_other_component(param1="value1")

# Use components together
result = component1.method(component2.prepare_input(input_data))
print(f"Integrated result: {result}")
```

### Error Handling
```python
# Import components and exceptions
from sifaka.component_type import create_component, ComponentError

# Create component
component = create_component(param1="value1")

# Handle potential errors
try:
    result = component.method(problematic_input)
except ComponentError as e:
    print(f"Error occurred: {e}")
    # Handle the error appropriately
    result = fallback_value
```

Attributes:
    attr1 (type): Description of attribute 1
    attr2 (type): Description of attribute 2
"""
```

## Component-Specific Example Guidelines

### Rules

Rule examples should demonstrate:
1. Creating the rule with different configurations
2. Validating text that passes and fails the rule
3. Accessing rule results and metadata
4. Integrating the rule with a chain

### Classifiers

Classifier examples should demonstrate:
1. Creating the classifier with different configurations
2. Classifying text with different characteristics
3. Accessing classification results and confidence scores
4. Batch classification of multiple texts
5. Integrating the classifier with a rule adapter

### Critics

Critic examples should demonstrate:
1. Creating the critic with different configurations
2. Critiquing text with different issues
3. Accessing critique results and suggestions
4. Integrating the critic with a chain
5. Customizing critic behavior with different prompts or parameters

### Model Providers

Model provider examples should demonstrate:
1. Creating the provider with different configurations
2. Generating text with different prompts
3. Controlling generation parameters (temperature, max tokens, etc.)
4. Handling rate limits and errors
5. Integrating the provider with a chain

### Chains

Chain examples should demonstrate:
1. Creating the chain with different components
2. Running the chain with different inputs
3. Handling chain results and metadata
4. Customizing chain behavior with different configurations
5. Integrating the chain with other systems

### Adapters

Adapter examples should demonstrate:
1. Creating the adapter with different configurations
2. Adapting external components to Sifaka interfaces
3. Using the adapter with Sifaka components
4. Handling adapter-specific configurations and behaviors

## Testing Examples

All examples in docstrings should be tested to ensure they work correctly. The Sifaka project includes a test suite for verifying docstring examples:

```python
# Run docstring tests
python -m pytest sifaka/tests/examples/test_api_documentation.py
```

When adding or updating examples, make sure to run these tests to verify that the examples work as expected.
