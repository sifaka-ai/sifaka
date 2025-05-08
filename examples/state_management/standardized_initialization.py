"""
Example of standardized initialization pattern for Sifaka components.

This module demonstrates the standard initialization pattern for Sifaka components,
showing both basic initialization in __init__() and resource-intensive initialization
in warm_up().
"""

from typing import Any, Dict, Optional, List

from pydantic import BaseModel, ConfigDict, PrivateAttr

from sifaka.utils.state import ComponentState, StateManager, create_state_manager


class ExampleComponentState(ComponentState):
    """State for example component."""

    # Configuration
    config: Dict[str, Any] = {}
    
    # Lightweight resources
    initialized_in_init: bool = False
    
    # Heavy resources
    model: Optional[Any] = None
    embeddings: Optional[List[float]] = None
    cache: Dict[str, Any] = {}


def create_example_state(**kwargs: Any) -> StateManager[ExampleComponentState]:
    """Create a state manager for an example component."""
    return create_state_manager(ExampleComponentState, **kwargs)


class StandardizedComponent(BaseModel):
    """
    Example component using standardized initialization pattern.
    
    This component demonstrates the standard initialization pattern:
    1. Basic initialization in __init__()
    2. Resource-intensive initialization in warm_up()
    """
    
    # Configuration (immutable)
    name: str
    description: str
    config: Dict[str, Any]
    
    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # State management
    _state_manager = PrivateAttr(default_factory=create_example_state)
    
    def __init__(
        self,
        name: str,
        description: str,
        config: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """
        Initialize the component with basic setup.
        
        This method performs lightweight initialization:
        - Stores configuration
        - Sets up basic state
        - Doesn't load heavy resources
        
        Args:
            name: Component name
            description: Component description
            config: Configuration dictionary
            **kwargs: Additional keyword arguments
        """
        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=config,
            **kwargs,
        )
        
        # Initialize state
        state = self._state_manager.get_state()
        state.initialized = False
        
        # Store configuration in state
        state.config = config
        
        # Perform lightweight initialization
        state.initialized_in_init = True
        
        # Mark basic initialization as complete
        state.initialized = True
        
        print(f"Basic initialization complete for {name}")
    
    def warm_up(self) -> None:
        """
        Initialize resource-intensive components.
        
        This method:
        - Checks if already initialized
        - Loads heavy resources like models
        - Handles initialization errors
        """
        state = self._state_manager.get_state()
        
        # Skip if already fully initialized
        if state.model is not None:
            print(f"Component {self.name} already warmed up")
            return
        
        print(f"Warming up {self.name}...")
        
        try:
            # Load heavy resources
            state.model = self._load_model()
            state.embeddings = self._load_embeddings()
            
            print(f"Component {self.name} successfully warmed up")
        except Exception as e:
            # Handle initialization errors
            error_msg = f"Failed to warm up {self.name}: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _load_model(self) -> Any:
        """
        Load a heavy model resource.
        
        This is a placeholder for actual model loading.
        
        Returns:
            A mock model object
        """
        print(f"Loading model for {self.name}...")
        # Simulate loading a heavy resource
        return {"type": "mock_model", "name": self.name}
    
    def _load_embeddings(self) -> List[float]:
        """
        Load embeddings.
        
        This is a placeholder for actual embeddings loading.
        
        Returns:
            A list of mock embeddings
        """
        print(f"Loading embeddings for {self.name}...")
        # Simulate loading embeddings
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def process(self, input_text: str) -> str:
        """
        Process input text using the component.
        
        This method demonstrates how to use state in a component method.
        
        Args:
            input_text: Text to process
            
        Returns:
            Processed text
            
        Raises:
            RuntimeError: If the component is not initialized
        """
        state = self._state_manager.get_state()
        
        # Check if initialized
        if not state.initialized:
            raise RuntimeError(f"Component {self.name} not initialized")
        
        # Check if model is loaded
        if state.model is None:
            # Auto-initialize if needed
            self.warm_up()
        
        # Check cache
        if input_text in state.cache:
            print(f"Using cached result for {input_text}")
            return state.cache[input_text]
        
        # Process input
        print(f"Processing {input_text} with {self.name}")
        result = f"Processed by {self.name}: {input_text}"
        
        # Cache result
        state.cache[input_text] = result
        
        return result


def main() -> None:
    """Run the example."""
    # Create component with basic initialization
    component = StandardizedComponent(
        name="example_component",
        description="An example component with standardized initialization",
        config={"param1": "value1", "param2": "value2"},
    )
    
    # Component is usable for basic operations after __init__
    print(f"Component name: {component.name}")
    print(f"Component description: {component.description}")
    
    # Warm up for resource-intensive operations
    component.warm_up()
    
    # Process input
    result = component.process("Hello, world!")
    print(f"Result: {result}")
    
    # Process again (should use cache)
    result = component.process("Hello, world!")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
