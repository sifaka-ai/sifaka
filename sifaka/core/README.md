# Sifaka Core

This package provides the foundation of the Sifaka framework, implementing core interfaces, components, and functionality that other modules build upon.

## Architecture

The core architecture follows a layered component-based design:

```
Core
├── Core Interfaces
│   ├── Component (base interface for all components)
│   ├── Configurable (interface for components with configuration)
│   ├── Stateful (interface for components with state management)
│   ├── Identifiable (interface for components with identity)
│   ├── Loggable (interface for components with logging)
│   └── Traceable (interface for components with tracing)
├── Base Components
│   ├── BaseComponent (foundation implementation for components)
│   ├── Generator (text generation component)
│   ├── Validator (validation component)
│   └── Improver (output improvement component)
├── Result Models
│   ├── BaseResult (foundation for all result types)
│   ├── RuleResult (results from rules)
│   ├── ValidationResult (results from validation)
│   ├── ClassificationResult (results from classification)
│   ├── ChainResult (results from chains)
│   └── CriticResult (results from critics)
└── Plugin System
    ├── Plugin (interface for plugin components)
    ├── PluginRegistry (discovers and registers plugins)
    └── PluginLoader (dynamically loads plugins)
```

## Core Components

- **Interfaces**: Define contracts that components must adhere to
- **Base Components**: Provide foundation implementations for standard components
- **Result Models**: Standardize result structures across the framework
- **Plugin System**: Enable extensibility through dynamic component loading
- **Protocol**: Define communication protocols between components
- **Dependency Management**: Handle component dependencies and injection

## Usage

### Using Core Interfaces

Core interfaces define the contracts that components must implement:

```python
from sifaka.core import Component, Configurable, Stateful

class MyComponent(Component, Configurable, Stateful):
    def __init__(self, name: str, config: dict = None):
        self._name = name
        self._config = config or {}
        self._state = {}

    def initialize(self) -> None:
        """Initialize component resources."""
        self._state["initialized"] = True

    def cleanup(self) -> None:
        """Clean up component resources."""
        self._state["initialized"] = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> dict:
        return self._config

    def update_config(self, config: dict) -> None:
        self._config.update(config)

    def get_state(self) -> dict:
        return self._state

    def set_state(self, state: dict) -> None:
        self._state.update(state)
```

### Using Generator

The Generator component provides standardized text generation:

```python
from sifaka.core.generation import Generator
from sifaka.models import OpenAIProvider

# Create model provider
model = OpenAIProvider("gpt-3.5-turbo")

# Create generator
generator = Generator(model=model)

# Generate text
output = generator.generate("Write a short story about a robot.")

# Access generator statistics
stats = generator.get_statistics()
print(f"Execution count: {stats['execution_count']}")
print(f"Average execution time: {stats['avg_execution_time']:.2f}s")

# Clear generator cache
generator.clear_cache()
```

### Using Validator

The Validator component provides standardized validation:

```python
from sifaka.core.validation import Validator, ValidatorConfig
from sifaka.rules import create_length_rule, create_profanity_rule

# Create validator with rules
validator = Validator(
    config=ValidatorConfig(
        rules=[
            create_length_rule(min_chars=10, max_chars=1000),
            create_profanity_rule()
        ],
        fail_fast=True
    )
)

# Validate output
result = validator.validate(
    input_value="Write a short story about friendship.",
    output_value="Friends are an important part of our lives. They provide support and companionship."
)

# Check validation result
if result.passed:
    print("Validation passed!")
else:
    print("Validation failed:")
    for rule_result in result.rule_results:
        if not rule_result.passed:
            print(f"- {rule_result.message}")
```

### Using Result Models

Result models provide standardized result structures:

```python
from sifaka.core.results import BaseResult, ValidationResult

# Create a simple result
result = BaseResult(
    passed=True,
    message="Operation completed successfully",
    metadata={"execution_time": 0.25}
)

# Create a validation result
validation_result = ValidationResult(
    passed=False,
    message="Validation failed",
    score=0.5,
    issues=["Text is too short", "Contains prohibited content"],
    suggestions=["Make the text longer", "Remove prohibited content"],
    metadata={"rule_id": "content_rule"}
)

# Access result properties
print(f"Passed: {result.passed}")
print(f"Message: {result.message}")
print(f"Metadata: {result.metadata}")
```

### Using the Plugin System

The plugin system enables dynamic component loading:

```python
from sifaka.core.plugins import PluginRegistry, PluginLoader

# Discover plugins
registry = PluginRegistry()
registry.discover_plugins("sifaka.plugins")

# Get plugin by name
plugin = registry.get_plugin("toxicity_classifier")

# Create component instance from plugin
classifier = plugin.create_component(config={"threshold": 0.7})

# Use the component
result = classifier.classify("This is a friendly message.")
print(f"Label: {result.label}, Confidence: {result.confidence:.2f}")
```

## Core Principles

The core module is built on several key principles:

1. **Composition over Inheritance**: Components are composed using interfaces rather than deep inheritance hierarchies
2. **Interface-Driven Design**: All components implement well-defined interfaces
3. **Standardized Results**: All operations return standardized result objects
4. **Error Handling**: Consistent error handling patterns across components
5. **State Management**: Standardized approach to state tracking and persistence
6. **Configurability**: All components can be configured through configuration objects
7. **Extensibility**: Plugin system enables adding new functionality without modifying core code

## Extending

### Creating a Custom Component

```python
from sifaka.core import Component
from sifaka.core.base import BaseComponent

class CustomProcessor(BaseComponent):
    """Custom text processing component."""

    def __init__(self, name: str = "custom_processor", **kwargs):
        super().__init__(name=name, **kwargs)
        # Initialize custom state or resources

    def process(self, text: str) -> str:
        """Process text using custom logic."""
        # Track execution
        self.increment_execution_count()

        # Process the text
        processed_text = text.upper()  # Simple example

        return processed_text

    def increment_execution_count(self):
        """Increment the execution count in state."""
        count = self._state_manager.get("execution_count", 0)
        self._state_manager.update("execution_count", count + 1)
```

### Creating a Custom Plugin

```python
from sifaka.core import Plugin
from typing import Any, Dict

class CustomPlugin(Plugin):
    @property
    def name(self) -> str:
        return "custom_plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def component_type(self) -> str:
        return "processor"

    def create_component(self, config: Dict[str, Any]) -> Any:
        """Create component instance from configuration."""
        from .custom_processor import CustomProcessor

        return CustomProcessor(
            name=config.get("name", "custom_processor"),
            description=config.get("description", "Custom text processor")
        )
```