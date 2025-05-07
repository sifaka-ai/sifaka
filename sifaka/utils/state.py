"""
State management utilities for Sifaka.

This module provides utility functions and classes for standardized state management
across the Sifaka framework. It includes classes for representing component state
and utilities for managing state in a consistent way.

## State Management

The module provides standardized state management:

1. **StateManager**: Utility class for managing component state
2. **ComponentState**: Base class for component state
3. **Specialized State Classes**: State classes for specific component types
   - **ClassifierState**: State for classifiers
   - **RuleState**: State for rules
   - **CriticState**: State for critics
   - **ModelState**: State for model providers
   - **ChainState**: State for chains
   - **AdapterState**: State for adapters

## State Creation

The module provides factory functions for creating state managers:

1. **create_state_manager**: Create a state manager for a specific state class
2. **create_classifier_state**: Create a state manager for a classifier
3. **create_rule_state**: Create a state manager for a rule
4. **create_critic_state**: Create a state manager for a critic
5. **create_model_state**: Create a state manager for a model provider
6. **create_chain_state**: Create a state manager for a chain
7. **create_adapter_state**: Create a state manager for an adapter

## Usage Examples

```python
from sifaka.utils.state import (
    create_classifier_state, ClassifierState, StateManager
)
from pydantic import BaseModel, PrivateAttr

# Create a component with state management
class MyClassifier(BaseModel):
    # Configuration
    name: str
    threshold: float = 0.5

    # State manager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize state
        self._state = self._state_manager.get_state()

    def warm_up(self) -> None:
        # Initialize the component
        if not self._state.initialized:
            self._state.model = self._load_model()
            self._state.cache = {}
            self._state.initialized = True

    def classify(self, text: str) -> str:
        # Use state in component methods
        if not self._state.initialized:
            self.warm_up()

        # Check cache
        if text in self._state.cache:
            return self._state.cache[text]

        # Use model
        result = self._state.model.predict(text)

        # Update cache
        self._state.cache[text] = result

        return result
```

## Direct State Pattern

In Pydantic v2, the recommended pattern is to use a direct state attribute:

```python
from sifaka.utils.state import ClassifierState
from pydantic import BaseModel, ConfigDict

class MyClassifier(BaseModel):
    # Configuration
    name: str
    threshold: float = 0.5

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize state directly
        self._state = ClassifierState()

    def warm_up(self) -> None:
        # Initialize the component
        if not self._state.initialized:
            self._state.model = self._load_model()
            self._state.cache = {}
            self._state.initialized = True
```
"""

from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, cast

from pydantic import BaseModel, PrivateAttr

T = TypeVar("T")


class StateManager(Generic[T]):
    """
    Utility class for standardized state management.

    This class provides a standardized way to manage state in Sifaka components.
    It handles initialization, access, and modification of state in a consistent way.

    Examples:
        ```python
        from sifaka.utils.state import StateManager

        class MyComponent(BaseModel):
            # Configuration
            name: str

            # State manager
            _state_manager = PrivateAttr(default_factory=create_classifier_state)

            def warm_up(self) -> None:
                # Initialize the component.
                state = self._state_manager.get_state()
                if not state.initialized:
                    state.model = self._load_model()
                    state.initialized = True
        ```
    """

    def __init__(
        self,
        initializer: Callable[[], T],
        initialized: bool = False,
        state: Optional[T] = None,
    ) -> None:
        """
        Initialize the state manager.

        Args:
            initializer: Function that initializes the state
            initialized: Whether the state is already initialized
            state: Initial state (if already initialized)
        """
        self._initializer = initializer
        self._initialized = initialized
        self._state = state

    def initialize(self) -> T:
        """
        Initialize the state if not already initialized.

        Returns:
            The initialized state
        """
        if not self._initialized:
            self._state = self._initializer()
            self._initialized = True
        return cast(T, self._state)

    def get_state(self) -> T:
        """
        Get the state, initializing it if necessary.

        Returns:
            The state
        """
        return self.initialize()

    def set_state(self, state: T) -> None:
        """
        Set the state.

        Args:
            state: The new state
        """
        self._state = state
        self._initialized = True

    def update_state(self, updater: Callable[[T], T]) -> T:
        """
        Update the state using an updater function.

        Args:
            updater: Function that takes the current state and returns the new state

        Returns:
            The updated state
        """
        state = self.get_state()
        new_state = updater(state)
        self.set_state(new_state)
        return new_state

    def reset(self) -> None:
        """Reset the state to uninitialized."""
        self._state = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the state is initialized."""
        return self._initialized


class ComponentState(BaseModel):
    """
    Base class for component state.

    This class provides a standardized way to represent component state.
    It can be extended to represent the state of specific component types.

    Examples:
        ```python
        from sifaka.utils.state import ComponentState, create_classifier_state

        class ClassifierState(ComponentState):
            model: Optional[Any] = None
            cache: Dict[str, Any] = {}
            feature_names: List[str] = []

        class MyClassifier(BaseModel):
            # Configuration
            name: str

            # State
            _state_manager = PrivateAttr(default_factory=create_classifier_state)

            def warm_up(self) -> None:
                # Initialize the classifier.
                state = self._state_manager.get_state()
                if not state.initialized:
                    state.model = self._load_model()
                    state.initialized = True
        ```
    """

    initialized: bool = False
    error: Optional[str] = None

    def reset(self) -> None:
        """Reset the state to uninitialized."""
        self.initialized = False
        self.error = None


class ClassifierState(ComponentState):
    """
    State for classifiers.

    This class represents the state of a classifier component.
    It includes common state variables used by classifiers.

    Examples:
        ```python
        from sifaka.utils.state import create_classifier_state

        class MyClassifier(BaseModel):
            # Configuration
            name: str

            # State
            _state_manager = PrivateAttr(default_factory=create_classifier_state)

            def warm_up(self) -> None:
                # Initialize the classifier.
                state = self._state_manager.get_state()
                if not state.initialized:
                    state.model = self._load_model()
                    state.initialized = True
        ```
    """

    model: Optional[Any] = None
    vectorizer: Optional[Any] = None
    pipeline: Optional[Any] = None
    feature_names: Dict[str, Any] = {}
    cache: Dict[str, Any] = {}
    dependencies_loaded: bool = False


class RuleState(ComponentState):
    """
    State for rules.

    This class represents the state of a rule component.
    It includes common state variables used by rules.

    Examples:
        ```python
        from sifaka.utils.state import create_rule_state

        class MyRule(BaseModel):
            # Configuration
            name: str

            # State
            _state_manager = PrivateAttr(default_factory=create_rule_state)

            def warm_up(self) -> None:
                # Initialize the rule.
                state = self._state_manager.get_state()
                if not state.initialized:
                    state.validator = self._create_validator()
                    state.initialized = True
        ```
    """

    validator: Optional[Any] = None
    handler: Optional[Any] = None
    cache: Dict[str, Any] = {}
    compiled_patterns: Dict[str, Any] = {}


class CriticState(ComponentState):
    """
    State for critics.

    This class represents the state of a critic component.
    It includes common state variables used by critics.

    Examples:
        ```python
        from sifaka.utils.state import create_critic_state

        class MyCritic(BaseModel):
            # Configuration
            name: str

            # State
            _state_manager = PrivateAttr(default_factory=create_critic_state)

            def warm_up(self) -> None:
                # Initialize the critic.
                state = self._state_manager.get_state()
                if not state.initialized:
                    state.model = self._load_model()
                    state.initialized = True
        ```
    """

    model: Optional[Any] = None
    prompt_manager: Optional[Any] = None
    response_parser: Optional[Any] = None
    memory_manager: Optional[Any] = None
    cache: Dict[str, Any] = {}


class ModelState(ComponentState):
    """
    State for model providers.

    This class represents the state of a model provider component.
    It includes common state variables used by model providers.

    Examples:
        ```python
        from sifaka.utils.state import create_model_state

        class MyModelProvider(BaseModel):
            # Configuration
            name: str

            # State
            _state_manager = PrivateAttr(default_factory=create_model_state)

            def warm_up(self) -> None:
                # Initialize the model provider.
                state = self._state_manager.get_state()
                if not state.initialized:
                    state.client = self._create_client()
                    state.initialized = True
        ```
    """

    client: Optional[Any] = None
    token_counter: Optional[Any] = None
    tracer: Optional[Any] = None
    cache: Dict[str, Any] = {}


class ChainState(ComponentState):
    """
    State for chains.

    This class represents the state of a chain component.
    It includes common state variables used by chains.

    Examples:
        ```python
        from sifaka.utils.state import create_chain_state

        class MyChain(BaseModel):
            # Configuration
            name: str

            # State
            _state_manager = PrivateAttr(default_factory=create_chain_state)

            def initialize(self) -> None:
                # Initialize the chain.
                state = self._state_manager.get_state()
                if not state.initialized:
                    state.generator = self._create_generator()
                    state.initialized = True
        ```
    """

    model: Optional[Any] = None
    generator: Optional[Any] = None
    validation_manager: Optional[Any] = None
    prompt_manager: Optional[Any] = None
    retry_strategy: Optional[Any] = None
    result_formatter: Optional[Any] = None
    critic: Optional[Any] = None
    cache: Dict[str, Any] = {}


class AdapterState(ComponentState):
    """
    State for adapters.

    This class represents the state of an adapter component.
    It includes common state variables used by adapters.

    Examples:
        ```python
        from sifaka.utils.state import create_adapter_state

        class MyAdapter(BaseModel):
            # Configuration
            name: str

            # State
            _state_manager = PrivateAttr(default_factory=create_adapter_state)

            def initialize(self) -> None:
                # Initialize the adapter.
                state = self._state_manager.get_state()
                if not state.initialized:
                    state.adaptee_cache = self._create_adaptee_cache()
                    state.initialized = True
        ```
    """

    adaptee: Optional[Any] = None
    adaptee_cache: Dict[str, Any] = {}
    config_cache: Dict[str, Any] = {}
    cache: Dict[str, Any] = {}


def create_state_manager(state_class: Type[T], **kwargs: Any) -> StateManager[T]:
    """
    Create a state manager for a specific state class.

    This function creates a StateManager instance for a specific state class,
    with an initializer that creates an instance of the state class with the
    provided arguments.

    Args:
        state_class: The state class to use
        **kwargs: Additional arguments to pass to the state class constructor

    Returns:
        A state manager for the specified state class

    Examples:
        ```python
        from sifaka.utils.state import create_state_manager, ComponentState

        # Create a custom state class
        class MyCustomState(ComponentState):
            model: Optional[Any] = None
            cache: Dict[str, Any] = {}

        # Create a state manager for the custom state class
        state_manager = create_state_manager(
            state_class=MyCustomState,
            initialized=False
        )

        # Use the state manager
        state = state_manager.get_state()
        state.model = load_model()
        state.initialized = True

        # Create with initial values
        state_manager = create_state_manager(
            state_class=MyCustomState,
            model=preloaded_model,
            initialized=True
        )
        ```
    """
    return StateManager(initializer=lambda: state_class(**kwargs))


def create_classifier_state(**kwargs: Any) -> StateManager[ClassifierState]:
    """
    Create a state manager for a classifier.

    This function creates a StateManager instance for a ClassifierState,
    with an initializer that creates a ClassifierState instance with the
    provided arguments.

    Args:
        **kwargs: Additional arguments to pass to the ClassifierState constructor

    Returns:
        A state manager for a classifier

    Examples:
        ```python
        from sifaka.utils.state import create_classifier_state
        from pydantic import BaseModel, PrivateAttr

        class MyClassifier(BaseModel):
            # Configuration
            name: str

            # State manager
            _state_manager = PrivateAttr(default_factory=create_classifier_state)

            def warm_up(self) -> None:
                # Initialize the classifier
                state = self._state_manager.get_state()
                if not state.initialized:
                    state.model = self._load_model()
                    state.cache = {}
                    state.initialized = True

            def classify(self, text: str) -> str:
                # Ensure initialized
                if not self._state_manager.is_initialized:
                    self.warm_up()

                # Get state
                state = self._state_manager.get_state()

                # Use model
                return state.model.predict(text)
        ```
    """
    return create_state_manager(ClassifierState, **kwargs)


def create_rule_state(**kwargs: Any) -> StateManager[RuleState]:
    """
    Create a state manager for a rule.

    Args:
        **kwargs: Additional arguments to pass to the RuleState constructor

    Returns:
        A state manager for a rule
    """
    return create_state_manager(RuleState, **kwargs)


def create_critic_state(**kwargs: Any) -> StateManager[CriticState]:
    """
    Create a state manager for a critic.

    Args:
        **kwargs: Additional arguments to pass to the CriticState constructor

    Returns:
        A state manager for a critic
    """
    return create_state_manager(CriticState, **kwargs)


def create_model_state(**kwargs: Any) -> StateManager[ModelState]:
    """
    Create a state manager for a model provider.

    Args:
        **kwargs: Additional arguments to pass to the ModelState constructor

    Returns:
        A state manager for a model provider
    """
    return create_state_manager(ModelState, **kwargs)


def create_chain_state(**kwargs: Any) -> StateManager[ChainState]:
    """
    Create a state manager for a chain.

    Args:
        **kwargs: Additional arguments to pass to the ChainState constructor

    Returns:
        A state manager for a chain
    """
    return create_state_manager(ChainState, **kwargs)


def create_adapter_state(**kwargs: Any) -> StateManager[AdapterState]:
    """
    Create a state manager for an adapter.

    Args:
        **kwargs: Additional arguments to pass to the AdapterState constructor

    Returns:
        A state manager for an adapter
    """
    return create_state_manager(AdapterState, **kwargs)
