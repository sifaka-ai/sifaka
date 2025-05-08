# Composition Over Inheritance Implementation for Chains

This document outlines the implementation of the Composition Over Inheritance pattern in the Sifaka chain system.

## Architecture

The chain system now uses composition over inheritance:
- `Chain` is the main class that delegates to specialized implementations
- `ChainImplementation` is a protocol that defines the core chain logic
- Specific implementations like `SimpleChainImplementation` and `BackoffChainImplementation` provide the actual functionality
- Factory functions like `create_simple_chain` and `create_backoff_chain` create `Chain` instances with the appropriate implementations

## Implementation Status

| Chain Type | Implementation Status | Factory Function Updated | Notes |
|------------|----------------------|-------------------------|-------|
| SimpleChain | ✅ Completed | ✅ Completed | Implemented with fixed retry strategy |
| BackoffChain | ✅ Completed | ✅ Completed | Implemented with exponential backoff retry strategy |

## Legacy Code Removal

The following legacy classes have been removed:
- `ChainCore` - Replaced by specific implementations of `ChainImplementation`
- `ChainOrchestrator` - Replaced by the `Chain` class

## Implementation Steps

### 1. ChainImplementation Protocol

The `ChainImplementation` protocol has been added to `implementation.py`:

```python
@runtime_checkable
class ChainImplementation(Protocol[OutputType]):
    """
    Protocol for chain implementations.

    This protocol defines the core chain logic that can be composed with
    the Chain class. It follows the composition over inheritance pattern,
    allowing for more flexible and maintainable code.
    """

    def run_impl(self, prompt: str) -> ChainResult[OutputType]:
        """
        Run the chain implementation with the given prompt.

        Args:
            prompt: The prompt to process

        Returns:
            The chain result with output, validation results, and critique details

        Raises:
            ValueError: If validation fails after max attempts
        """
        ...

    def warm_up_impl(self) -> None:
        """
        Warm up the chain implementation.

        This method initializes any resources needed by the chain implementation.
        """
        ...
```

### 2. Chain Class

The `Chain` class has been added to `implementation.py`:

```python
class Chain(BaseModel, Generic[OutputType]):
    """
    Chain that uses composition over inheritance.

    This class delegates chain execution to an implementation object
    rather than using inheritance. It follows the composition over inheritance
    pattern to create a more flexible and maintainable design.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Name of the chain", min_length=1)
    description: str = Field(description="Description of the chain", min_length=1)
    config: ChainConfig = Field(description="Configuration for the chain")
    _implementation: ChainImplementation[OutputType] = PrivateAttr()
    _state_manager: StateManager[ChainState] = PrivateAttr(default_factory=create_chain_state)

    def __init__(
        self,
        name: str,
        description: str,
        config: ChainConfig,
        implementation: ChainImplementation[OutputType],
        **kwargs: Any,
    ):
        """Initialize the chain."""
        super().__init__(name=name, description=description, config=config, **kwargs)
        self._implementation = implementation

    def run(self, prompt: str) -> ChainResult[OutputType]:
        """Run the chain with the given prompt."""
        return self._implementation.run_impl(prompt)

    def warm_up(self) -> None:
        """Warm up the chain."""
        self._implementation.warm_up_impl()
```

### 3. Implementation Plan for Each Chain Type

For each chain type, we need to:

1. Create an implementation class that follows the `ChainImplementation` protocol
2. Move the core logic from the chain to the implementation class
3. Update the factory function to create a `Chain` with the implementation

#### 3.1 SimpleChainImplementation

```python
class SimpleChainImplementation(BaseModel, Generic[OutputType]):
    """
    Simple chain implementation that uses a fixed number of retries.

    This class implements the ChainImplementation protocol for a simple chain
    with a fixed number of retries. It follows the composition over inheritance
    pattern, allowing it to be composed with the Chain class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_chain_state)

    def __init__(
        self,
        model: ModelProvider,
        rules: List[Rule],
        critic: Optional[CriticCore] = None,
        max_attempts: int = 3,
        **kwargs: Any,
    ):
        """Initialize a SimpleChainImplementation instance."""
        # Initialize the base class
        super().__init__(**kwargs)

        # Create components
        validation_manager = ValidationManager[OutputType](rules)
        prompt_manager = PromptManager()
        retry_strategy = SimpleRetryStrategy[OutputType](max_attempts=max_attempts)
        result_formatter = ResultFormatter[OutputType]()

        # Initialize state
        state = self._state_manager.get_state()
        state.initialized = False

        # Store components in state
        state.model = model
        state.validation_manager = validation_manager
        state.prompt_manager = prompt_manager
        state.retry_strategy = retry_strategy
        state.result_formatter = result_formatter
        state.critic = critic

        # Create generator
        state.generator = Generator[OutputType](model)

        # Mark as initialized
        state.initialized = True

    def run_impl(self, prompt: str) -> ChainResult[OutputType]:
        """Run the chain implementation with the given prompt."""
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("SimpleChainImplementation not properly initialized")

        # Delegate to retry strategy
        return state.retry_strategy.run(
            prompt=prompt,
            generator=state.generator,
            validation_manager=state.validation_manager,
            prompt_manager=state.prompt_manager,
            result_formatter=state.result_formatter,
            critic=state.critic,
        )

    def warm_up_impl(self) -> None:
        """Warm up the chain implementation."""
        # Currently, there's nothing to warm up
        pass
```

#### 3.2 BackoffChainImplementation

```python
class BackoffChainImplementation(BaseModel, Generic[OutputType]):
    """
    Backoff chain implementation that uses exponential backoff for retries.

    This class implements the ChainImplementation protocol for a chain with
    exponential backoff retry strategy. It follows the composition over inheritance
    pattern, allowing it to be composed with the Chain class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_chain_state)

    def __init__(
        self,
        model: ModelProvider,
        rules: List[Rule],
        critic: Optional[CriticCore] = None,
        max_attempts: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        max_backoff: float = 60.0,
        **kwargs: Any,
    ):
        """Initialize a BackoffChainImplementation instance."""
        # Initialize the base class
        super().__init__(**kwargs)

        # Create components
        validation_manager = ValidationManager[OutputType](rules)
        prompt_manager = PromptManager()
        retry_strategy = BackoffRetryStrategy[OutputType](
            max_attempts=max_attempts,
            initial_backoff=initial_backoff,
            backoff_factor=backoff_factor,
            max_backoff=max_backoff,
        )
        result_formatter = ResultFormatter[OutputType]()

        # Initialize state
        state = self._state_manager.get_state()
        state.initialized = False

        # Store components in state
        state.model = model
        state.validation_manager = validation_manager
        state.prompt_manager = prompt_manager
        state.retry_strategy = retry_strategy
        state.result_formatter = result_formatter
        state.critic = critic

        # Create generator
        state.generator = Generator[OutputType](model)

        # Mark as initialized
        state.initialized = True

    def run_impl(self, prompt: str) -> ChainResult[OutputType]:
        """Run the chain implementation with the given prompt."""
        state = self._state_manager.get_state()
        if not state.initialized:
            raise RuntimeError("BackoffChainImplementation not properly initialized")

        # Delegate to retry strategy
        return state.retry_strategy.run(
            prompt=prompt,
            generator=state.generator,
            validation_manager=state.validation_manager,
            prompt_manager=state.prompt_manager,
            result_formatter=state.result_formatter,
            critic=state.critic,
        )

    def warm_up_impl(self) -> None:
        """Warm up the chain implementation."""
        # Currently, there's nothing to warm up
        pass
```

### 4. Update Factory Functions

The factory functions have been updated to create a `Chain` with the implementation:

```python
def create_simple_chain(
    model: ModelProvider,
    rules: List[Rule],
    critic: Optional[CriticCore] = None,
    max_attempts: int = 3,
    name: str = "simple_chain",
    description: str = "A simple chain with a fixed number of retries",
) -> Chain[OutputType]:
    """Create a simple chain with the given parameters."""
    # Create config
    config = ChainConfig(
        max_attempts=max_attempts,
        params={
            "use_critic": critic is not None,
        },
    )

    # Create implementation
    implementation = SimpleChainImplementation[OutputType](
        model=model,
        rules=rules,
        critic=critic,
        max_attempts=max_attempts,
    )

    # Create and return chain
    return Chain[OutputType](
        name=name,
        description=description,
        config=config,
        implementation=implementation,
    )
```

## Benefits

- Reduced complexity by avoiding deep inheritance hierarchies
- Improved flexibility by allowing components to be combined in different ways
- Better testability by enabling testing of implementations in isolation
- Reduced coupling between components
- More consistent API across all chain types

## Next Steps After Implementation

1. Update tests to work with the new pattern:
   - Update test fixtures to create chains using the new pattern
   - Update assertions to work with the new chain structure

2. Update documentation to reflect the new pattern:
   - Update docstrings to describe the new architecture
   - Update examples to use the new factory functions
   - Add implementation notes about the Composition Over Inheritance pattern
