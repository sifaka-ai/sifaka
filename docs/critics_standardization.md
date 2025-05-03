# Critics Standardization

This document outlines the standardization of critics in the Sifaka codebase.

## Component-Based Architecture

All critics in Sifaka follow a component-based architecture where functionality is delegated to specialized components:

1. **Core Components**:
   - `CriticCore`: Main implementation that delegates to specialized components
   - `BaseCritic`: Abstract base class for critics

2. **Managers**:
   - `PromptManager`: Creates prompts for validation, critique, improvement, and reflection
   - `ResponseParser`: Parses responses from language models
   - `MemoryManager`: Manages memory for critics (used by ReflexionCritic)

3. **Services**:
   - `CritiqueService`: Provides methods for validation, critique, and improvement

## Factory Function Patterns

All factory functions in Sifaka follow a consistent pattern:

```python
def create_X_critic(
    llm_provider: Any,
    name: str = "x_critic",
    description: str = "Description of X critic",
    min_confidence: float = 0.7,
    max_attempts: int = 3,
    cache_size: int = 100,
    priority: int = 1,
    cost: float = 1.0,
    # Additional parameters specific to the critic type
    config: CriticConfig = None,
    **kwargs: Any,
) -> CriticCore:
    """
    Create an X critic with the given parameters.
    
    This factory function creates a configured X critic instance.
    
    Args:
        llm_provider: Language model provider to use
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache
        priority: Priority of the critic
        cost: Cost of using the critic
        # Additional parameters specific to the critic type
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic
        
    Returns:
        A configured X critic
    """
    # Use provided config or create one from parameters
    if config is None:
        config = XCriticConfig(
            name=name,
            description=description,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            # Additional parameters specific to the critic type
        )
        
    # Create managers
    prompt_manager = XCriticPromptManager(config)
    response_parser = ResponseParser()
    # Additional managers specific to the critic type
    
    # Create critic
    return CriticCore(
        config=config,
        llm_provider=llm_provider,
        prompt_manager=prompt_manager,
        response_parser=response_parser,
        # Additional managers specific to the critic type
        **kwargs,
    )
```

## Memory Management

Memory management in Sifaka follows a consistent pattern:

1. **Memory Initialization**:
   ```python
   memory_manager = MemoryManager(buffer_size=memory_buffer_size)
   ```

2. **Memory Usage**:
   ```python
   # Add to memory
   memory_manager.add_to_memory(item)
   
   # Get from memory
   items = memory_manager.get_memory(max_items=5)
   
   # Clear memory
   memory_manager.clear_memory()
   ```

## Creating a Custom Critic

To create a custom critic, follow these steps:

1. **Create a Custom Prompt Manager**:
   ```python
   from sifaka.critics.managers import PromptManager
   
   class CustomPromptManager(PromptManager):
       """Custom prompt manager for your critic."""
       
       def _create_validation_prompt_impl(self, text: str) -> str:
           """Create a validation prompt."""
           # Your implementation here
           
       def _create_critique_prompt_impl(self, text: str) -> str:
           """Create a critique prompt."""
           # Your implementation here
           
       def _create_improvement_prompt_impl(
           self, text: str, feedback: str, reflections: Optional[List[str]] = None
       ) -> str:
           """Create an improvement prompt."""
           # Your implementation here
           
       def _create_reflection_prompt_impl(
           self, original_text: str, feedback: str, improved_text: str
       ) -> str:
           """Create a reflection prompt."""
           # Your implementation here
   ```

2. **Create a Custom Config**:
   ```python
   from sifaka.critics.models import CriticConfig
   
   class CustomCriticConfig(CriticConfig):
       """Configuration for your critic."""
       
       custom_param: str = Field(
           default="default_value",
           description="Description of custom parameter",
       )
   ```

3. **Create a Factory Function**:
   ```python
   def create_custom_critic(
       llm_provider: Any,
       name: str = "custom_critic",
       description: str = "Custom critic implementation",
       min_confidence: float = 0.7,
       max_attempts: int = 3,
       cache_size: int = 100,
       priority: int = 1,
       cost: float = 1.0,
       custom_param: str = "default_value",
       config: CriticConfig = None,
       **kwargs: Any,
   ) -> CriticCore:
       """Create a custom critic."""
       # Use provided config or create one from parameters
       if config is None:
           config = CustomCriticConfig(
               name=name,
               description=description,
               min_confidence=min_confidence,
               max_attempts=max_attempts,
               cache_size=cache_size,
               priority=priority,
               cost=cost,
               custom_param=custom_param,
           )
           
       # Create managers
       prompt_manager = CustomPromptManager(config)
       response_parser = ResponseParser()
       
       # Create critic
       return CriticCore(
           config=config,
           llm_provider=llm_provider,
           prompt_manager=prompt_manager,
           response_parser=response_parser,
           **kwargs,
       )
   ```

4. **Use Your Custom Critic**:
   ```python
   from sifaka.models import OpenAIProvider
   
   # Create a model provider
   model = OpenAIProvider(model="gpt-4")
   
   # Create your custom critic
   critic = create_custom_critic(
       llm_provider=model,
       name="my_critic",
       description="My custom critic",
       custom_param="custom_value",
   )
   
   # Use your critic
   result = critic.critique("This is a test.")
   ```
