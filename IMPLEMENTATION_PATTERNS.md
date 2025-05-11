# Standardized Implementation Patterns for Sifaka

This document defines the standardized implementation patterns to be used across all components in the Sifaka framework. Following these patterns ensures consistency, maintainability, and extensibility of the codebase.

## 1. Component Lifecycle Management

All components should implement a consistent lifecycle with the following phases:

### 1.1 Initialization Phase

```python
def __init__(self, name: str, description: str, config: Optional[ComponentConfig] = None, **kwargs: Any):
    """Initialize the component."""
    # Create config if not provided
    if config is None:
        config = ComponentConfig(name=name, description=description, **kwargs)
    
    # Initialize state manager
    self._state_manager = StateManager()
    
    # Set basic state
    self._state_manager.update("initialized", False)
    self._state_manager.update("cache", {})
    
    # Set metadata
    self._state_manager.set_metadata("component_type", self.__class__.__name__)
    self._state_manager.set_metadata("creation_time", time.time())
```

### 1.2 Warm-up Phase

```python
def warm_up(self) -> None:
    """Prepare the component for use."""
    try:
        # Check if already initialized
        if self._state_manager.get("initialized", False):
            logger.debug(f"Component {self.name} already initialized")
            return
            
        # Initialize resources
        self._initialize_resources()
        
        # Mark as initialized
        self._state_manager.update("initialized", True)
        self._state_manager.set_metadata("warm_up_time", time.time())
        
        logger.debug(f"Component {self.name} warmed up successfully")
        
    except Exception as e:
        self.record_error(e)
        raise InitializationError(f"Failed to warm up component {self.name}: {str(e)}") from e
```

### 1.3 Operation Phase

```python
def process(self, input: T) -> R:
    """Process the input and return a result."""
    # Ensure component is initialized
    if not self._state_manager.get("initialized", False):
        self.warm_up()
    
    # Process input
    start_time = time.time()
    
    # Define the operation
    def operation():
        # Actual processing logic
        result = self._process_input(input)
        return result
    
    # Use standardized error handling
    result = safely_execute_component_operation(
        operation=operation,
        component_name=self.name,
        component_type=self.__class__.__name__,
        additional_metadata={"input_type": type(input).__name__}
    )
    
    # Update statistics
    processing_time = time.time() - start_time
    self.update_statistics(result, processing_time_ms=processing_time * 1000)
    
    return result
```

### 1.4 Cleanup Phase

```python
def cleanup(self) -> None:
    """Clean up component resources."""
    try:
        # Release resources
        self._release_resources()
        
        # Clear cache
        if hasattr(self, "clear_cache") and callable(getattr(self, "clear_cache")):
            self.clear_cache()
        
        # Reset initialization flag
        self._state_manager.update("initialized", False)
        
        logger.debug(f"Component {self.name} cleaned up successfully")
        
    except Exception as e:
        # Log but don't raise
        logger.error(f"Failed to clean up component {self.name}: {str(e)}")
```

## 2. Factory Function Pattern

All factory functions should follow this standardized pattern:

```python
def create_component(
    # Required parameters
    name: str,
    # Optional parameters with defaults
    description: Optional[str] = None,
    # Configuration parameter
    config: Optional[Union[Dict[str, Any], ComponentConfig]] = None,
    # Component-specific parameters
    **kwargs: Any,
) -> Component:
    """
    Create a component with the given configuration.
    
    Args:
        name: Name of the component
        description: Optional description of the component
        config: Optional configuration (either a dict or ComponentConfig)
        **kwargs: Additional parameters for the component
        
    Returns:
        Configured component instance
        
    Raises:
        ValueError: If configuration is invalid
        TypeError: If input types are incompatible
    """
    try:
        # Set default description if not provided
        description = description or f"{name.title()} component"
        
        # Standardize configuration
        config = standardize_component_config(
            config=config,
            name=name,
            description=description,
            **kwargs
        )
        
        # Create component
        component = Component(
            name=name,
            description=description,
            config=config
        )
        
        # Initialize if needed
        component.warm_up()
        
        return component
        
    except Exception as e:
        logger.error(f"Error creating component: {e}")
        raise ValueError(f"Error creating component: {str(e)}")
```

## 3. Pattern Matching Utilities

All components should use the standardized pattern matching utilities from `utils/patterns.py`:

```python
# INCORRECT - Using raw regex
import re
pattern = re.compile(r"\s+")
normalized = re.sub(pattern, " ", text)

# CORRECT - Using standardized utilities
from sifaka.utils.patterns import compile_pattern, replace_pattern
pattern = compile_pattern(r"\s+")
normalized = replace_pattern(text, pattern, " ")
```

## 4. Error Handling Pattern

All components should use the standardized error handling patterns from `utils/error_patterns.py`:

```python
# INCORRECT - Custom try/except
try:
    result = self._process(input)
    return result
except Exception as e:
    logger.error(f"Error processing input: {e}")
    return ErrorResult(error=str(e))

# CORRECT - Using standardized error handling
from sifaka.utils.error_patterns import safely_execute_component_operation

def operation():
    return self._process(input)

result = safely_execute_component_operation(
    operation=operation,
    component_name=self.name,
    component_type=self.__class__.__name__,
    additional_metadata={"input_type": type(input).__name__}
)
```

## 5. State Management Pattern

All components should use the standardized state management from `utils/state.py`:

```python
# INCORRECT - Custom state management
self._state = {}
self._state["initialized"] = False
self._state["cache"] = {}

# CORRECT - Using standardized state management
from sifaka.utils.state import StateManager
self._state_manager = StateManager()
self._state_manager.update("initialized", False)
self._state_manager.update("cache", {})
```

## 6. Configuration Management Pattern

All components should use the standardized configuration management from `utils/config.py`:

```python
# INCORRECT - Custom configuration
self.config = config or {}
self.temperature = self.config.get("temperature", 0.7)

# CORRECT - Using standardized configuration
from sifaka.utils.config import standardize_component_config
self.config = standardize_component_config(config=config, **kwargs)
```

## 7. Documentation Pattern

All components should follow the standardized documentation pattern:

```python
"""
Component Name

Brief description of the component.

## Overview
Detailed explanation of the component's role in the system.

## Architecture
Description of the component's architecture and design patterns.

## Lifecycle
Description of the component's lifecycle (initialization, operation, cleanup).

## Error Handling
Description of how the component handles errors and exceptions.

## Examples
Code examples showing common usage patterns.
"""
```

## Implementation Priority

1. Core components (BaseComponent, etc.)
2. Model providers
3. Rules and validators
4. Critics
5. Chain components
6. Retrieval components
7. Adapters
8. Classifiers
