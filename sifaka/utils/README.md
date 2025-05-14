# Sifaka Utils

This package provides foundational utilities and helper functions that power the entire Sifaka framework. These utilities implement common patterns, standardized configurations, state management, error handling, logging, and other cross-cutting concerns.

## Architecture

The utils architecture follows a modular design with specialized components for different concerns:

```
Utils
├── State Management
│   ├── StateManager (centralized state tracking)
│   ├── State (immutable state container)
│   └── Component States (specialized states for components)
├── Configuration
│   ├── BaseConfig (foundation for all configs)
│   ├── Component Configs (chain, models, rules, etc.)
│   └── Configuration Factories (standardized creation)
├── Logging
│   ├── EnhancedLogger (extended logging capabilities)
│   ├── StructuredFormatter (structured log formatting)
│   └── LoggerFactory (standardized logger creation)
├── Error Handling
│   ├── Base Exceptions (foundation for error types)
│   ├── Component Errors (specialized error classes)
│   └── Error Formatting (standardized error messaging)
├── Result Types
│   ├── BaseResult (foundation for result types)
│   ├── Component Results (specialized result structures)
│   └── Result Factories (standardized creation)
└── Common Utilities
    ├── Text Processing (string manipulation)
    ├── Resource Management (file and web resources)
    ├── Tracing (execution and performance tracing)
    └── Pattern Implementations (reusable design patterns)
```

## Core Components

### State Management

The state management module provides a standardized way to track, update, and manage component state throughout the Sifaka framework.

```python
from sifaka.utils.state import StateManager

# Create a state manager
state_manager = StateManager()

# Update state
state_manager.update("initialized", True)
state_manager.update("cache", {})

# Read state
is_initialized = state_manager.get("initialized", False)
cache = state_manager.get("cache", {})

# Set metadata
state_manager.set_metadata("component_type", "Validator")

# Rollback to previous state
state_manager.rollback()

# Reset state
state_manager.reset()
```

For specialized components, factory functions create state managers with appropriate defaults:

```python
from sifaka.utils.state import create_classifier_state, create_chain_state

# Create specialized state managers
classifier_state = create_classifier_state(
    initialized=True,
    model=my_model,
    vectorizer=my_vectorizer
)

chain_state = create_chain_state(
    model=my_model,
    validation_manager=validation_manager,
    prompt_manager=prompt_manager
)
```

### Logging

The logging module provides enhanced logging capabilities with structured logging, operation tracking, and standardized formatting.

```python
from sifaka.utils.logging import get_logger

# Get a logger
logger = get_logger("my_component")

# Basic logging
logger.info("Processing started")
logger.error("An error occurred: %s", error_message)
logger.warning("Resource nearly depleted: %d remaining", remaining)

# Enhanced logging
logger.success("Operation completed successfully")

# Structured logging
logger.structured(
    logger.INFO,
    "Processing data",
    data_size=1024,
    processing_time=0.5,
    status="success"
)

# Operation tracking
with logger.operation_context("data_processing"):
    # Operation code
    process_data()
    # Automatically logs start, end, and timing
```

### Configuration

The configuration module provides standardized configuration for all Sifaka components with validation and normalization.

```python
from sifaka.utils.config.rules import RuleConfig, standardize_rule_config
from sifaka.utils.config.models import ModelConfig
from sifaka.utils.config.chain import ChainConfig

# Create configs with validation
rule_config = RuleConfig(
    name="length_rule",
    description="Validates text length",
    priority="HIGH",
    params={"min_chars": 10, "max_chars": 1000}
)

# Standardize configs from dict input
raw_config = {
    "name": "prohibited_content",
    "params": {"terms": ["bad", "offensive"]}
}
standardized = standardize_rule_config(raw_config)

# Create model config
model_config = ModelConfig(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# Create chain config
chain_config = ChainConfig(
    max_attempts=3,
    timeout=30,
    fail_fast=True
)
```

### Results

The results module provides standardized result types for all Sifaka components with factories for consistent creation.

```python
from sifaka.utils.results import (
    create_rule_result,
    create_error_result,
    create_classification_result
)

# Create a rule validation result
result = create_rule_result(
    passed=True,
    message="Content length is acceptable",
    score=1.0
)

# Create an error result
error_result = create_error_result(
    error="Invalid input",
    code="INVALID_INPUT",
    details={"input_type": "str", "expected_type": "dict"}
)

# Create a classification result
classification = create_classification_result(
    label="positive",
    confidence=0.87,
    metadata={"model": "sentiment-analyzer"}
)
```

### Text Processing

The text module provides utilities for text manipulation, tokenization, and processing.

```python
from sifaka.utils.text import (
    truncate_text,
    count_tokens,
    detect_language,
    normalize_text
)

# Truncate text to specified length
truncated = truncate_text("This is a very long text...", max_length=20)

# Count tokens using specified tokenizer
token_count = count_tokens("How many tokens is this?", tokenizer="cl100k_base")

# Detect language
language = detect_language("Bonjour le monde!")  # Returns 'fr'

# Normalize text (lowercase, remove extra whitespace)
normalized = normalize_text("  Text   with EXTRA   spaces  ")
```

### Error Handling

The errors module provides standardized error classes and handling patterns for all Sifaka components.

```python
from sifaka.utils.errors import (
    ValidationError,
    ModelError,
    ConfigurationError,
    handle_errors
)

# Using error classes
try:
    # Some operation
    if invalid_input:
        raise ValidationError("Invalid input", code="INVALID_INPUT")
except ValidationError as e:
    print(f"Validation error: {e.message}, code: {e.code}")

# Using error handler decorator
@handle_errors(default_value=None, logger=logger)
def process_data(data):
    # Process data with automatic error handling
    return result
```

### Tracing

The tracing module provides utilities for performance monitoring and execution tracing.

```python
from sifaka.utils.tracing import Tracer, trace

# Create a tracer
tracer = Tracer(name="data_processor")

# Trace a function call
with tracer.trace_span("process_data"):
    # Function code
    process_data()

# Use the trace decorator
@trace(name="validate_content")
def validate_content(text):
    # Validation code
    return result
```

## Using Utils in Components

Utils provide the foundation for implementing Sifaka components:

```python
from sifaka.utils.state import create_rule_state
from sifaka.utils.logging import get_logger
from sifaka.utils.results import create_rule_result
from pydantic import BaseModel, PrivateAttr

class MyCustomRule(BaseModel):
    """A custom rule implementation using Sifaka utils."""

    name: str
    threshold: float = 0.5

    # Private utilities
    _logger = PrivateAttr(default_factory=lambda: get_logger("custom_rule"))
    _state = PrivateAttr(default_factory=create_rule_state)

    def __init__(self, **data):
        super().__init__(**data)
        self._state.update("initialized", True)
        self._logger.info("Rule initialized with threshold: %f", self.threshold)

    def validate(self, text):
        """Validate text using this rule."""
        # Access state
        cache = self._state.get("cache", {})

        # Check cache
        if text in cache:
            self._logger.info("Using cached result")
            return cache[text]

        # Custom validation logic
        score = self._calculate_score(text)
        passed = score >= self.threshold

        # Create standardized result
        result = create_rule_result(
            passed=passed,
            message=f"Score {score} is {'above' if passed else 'below'} threshold {self.threshold}",
            score=score,
            issues=[] if passed else ["Score below threshold"],
            suggestions=[] if passed else ["Improve content to increase score"]
        )

        # Update cache
        cache[text] = result
        self._state.update("cache", cache)

        return result
```

## Best Practices

1. **Use factory functions**: Create utilities using provided factory functions
2. **Standardize configuration**: Use config classes for component configuration
3. **Manage state properly**: Use StateManager for all stateful components
4. **Use structured logging**: Leverage structured logging for better observability
5. **Standardize results**: Use result factory functions for consistent responses
6. **Handle errors gracefully**: Use standardized error classes and handlers
7. **Trace performance**: Use tracing utilities to monitor execution
8. **Follow patterns**: Implement common design patterns consistently
9. **Keep utilities pure**: Avoid side effects in utility functions
10. **Document usage**: Add clear examples in docstrings for all utilities