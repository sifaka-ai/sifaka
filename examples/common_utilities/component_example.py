"""
Example Component Using Common Utilities

This example demonstrates how to use the common utilities in a component.
It shows the standardized implementation pattern for Sifaka components.

Key concepts:
1. Use _state_manager for state management
2. Use common utilities for state access, error handling, and result creation
3. Follow the standardized component implementation pattern
4. Separate configuration from state
5. Handle errors consistently
"""

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from sifaka.utils.state import StateManager
from sifaka.utils.common import (
    initialize_component_state,
    get_cached_result,
    update_cache,
    update_statistics,
    record_error,
    safely_execute,
    create_standard_result
)


class ExampleComponentConfig(BaseModel):
    """Configuration for the example component."""
    
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for processing"
    )
    cache_size: int = Field(
        default=100,
        ge=0,
        description="Maximum number of cached results"
    )
    model_config = ConfigDict(frozen=True)


class ExampleComponent(BaseModel):
    """
    Example component using common utilities.
    
    This component demonstrates the standardized implementation pattern
    for Sifaka components, including state management, error handling,
    and result creation.
    """
    
    name: str = Field(
        default="example_component",
        description="Name of the component"
    )
    description: str = Field(
        default="Example component using common utilities",
        description="Description of the component"
    )
    config: ExampleComponentConfig = Field(
        default_factory=ExampleComponentConfig,
        description="Component configuration"
    )
    
    # Private state manager
    _state_manager = PrivateAttr(default_factory=StateManager)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize state
        initialize_component_state(
            self._state_manager,
            "ExampleComponent",
            self.name,
            self.description
        )
    
    def process(self, input_data: str) -> Dict[str, Any]:
        """
        Process input data using the standardized pattern.
        
        Args:
            input_data: The input data to process
            
        Returns:
            A standardized result dictionary
        """
        # Check cache
        cache_key = str(input_data)
        cached_result = get_cached_result(self._state_manager, cache_key)
        if cached_result:
            return cached_result
        
        # Process data
        start_time = time.time()
        result = safely_execute(
            lambda: self._process_impl(input_data),
            component_name=self.name,
            state_manager=self._state_manager,
            default_value=None
        )
        
        # Update statistics
        execution_time = time.time() - start_time
        update_statistics(
            self._state_manager,
            execution_time,
            success=(result is not None)
        )
        
        # Create result
        final_result = create_standard_result(
            output=result,
            success=(result is not None),
            metadata={"input_length": len(input_data)},
            processing_time_ms=execution_time * 1000
        )
        
        # Update cache
        update_cache(
            self._state_manager,
            cache_key,
            final_result,
            max_size=self.config.cache_size
        )
        
        return final_result
    
    def _process_impl(self, input_data: str) -> Any:
        """
        Implementation-specific processing.
        
        Args:
            input_data: The input data to process
            
        Returns:
            The processed data
            
        Raises:
            ValueError: If the input data is invalid
        """
        # Simulate processing
        if not input_data:
            raise ValueError("Input data cannot be empty")
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Return processed data
        return {
            "processed": input_data.upper(),
            "length": len(input_data),
            "above_threshold": len(input_data) > self.config.threshold * 10
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get component statistics.
        
        Returns:
            A dictionary of component statistics
        """
        return {
            "execution_count": self._state_manager.get("execution_count", 0),
            "success_count": self._state_manager.get("success_count", 0),
            "error_count": self._state_manager.get("error_count", 0),
            "cache_hits": self._state_manager.get("cache_hits", 0),
            "avg_execution_time_ms": self._state_manager.get_metadata("avg_execution_time_ms", 0),
            "last_cache_hit": self._state_manager.get_metadata("last_cache_hit", None),
            "last_cache_update": self._state_manager.get_metadata("last_cache_update", None),
            "last_error": self._state_manager.get_metadata("last_error", None)
        }
    
    def clear_cache(self) -> None:
        """Clear the component cache."""
        self._state_manager.update("result_cache", {})
        self._state_manager.update("cache_hits", 0)


# Example usage
if __name__ == "__main__":
    # Create component
    component = ExampleComponent(
        name="my_example",
        description="My example component",
        config=ExampleComponentConfig(threshold=0.7, cache_size=50)
    )
    
    # Process data
    print("Processing 'hello'...")
    result1 = component.process("hello")
    print(f"Result: {result1}")
    
    # Process same data again (should use cache)
    print("\nProcessing 'hello' again...")
    result2 = component.process("hello")
    print(f"Result: {result2}")
    
    # Process different data
    print("\nProcessing 'world'...")
    result3 = component.process("world")
    print(f"Result: {result3}")
    
    # Try processing invalid data
    print("\nProcessing empty string...")
    result4 = component.process("")
    print(f"Result: {result4}")
    
    # Get statistics
    print("\nComponent statistics:")
    stats = component.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
