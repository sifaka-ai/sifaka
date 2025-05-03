"""
State management utilities for Sifaka.

This module provides utility functions and classes for standardized state management
across the Sifaka framework.
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
            _state = PrivateAttr(default_factory=lambda: StateManager(
                initializer=lambda: {"model": None, "cache": {}}
            ))
            
            def warm_up(self) -> None:
                """Initialize the component."""
                self._state.initialize()
                
            def get_state(self) -> Dict[str, Any]:
                """Get the component's state."""
                return self._state.get_state()
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
        from sifaka.utils.state import ComponentState

        class ClassifierState(ComponentState):
            model: Optional[Any] = None
            cache: Dict[str, Any] = {}
            feature_names: List[str] = []
            
        class MyClassifier(BaseModel):
            # Configuration
            name: str
            
            # State
            _state: ClassifierState = PrivateAttr(default_factory=ClassifierState)
            
            def warm_up(self) -> None:
                """Initialize the classifier."""
                if not self._state.initialized:
                    self._state.model = self._load_model()
                    self._state.initialized = True
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
        from sifaka.utils.state import ClassifierState

        class MyClassifier(BaseModel):
            # Configuration
            name: str
            
            # State
            _state: ClassifierState = PrivateAttr(default_factory=ClassifierState)
            
            def warm_up(self) -> None:
                """Initialize the classifier."""
                if not self._state.initialized:
                    self._state.model = self._load_model()
                    self._state.initialized = True
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
        from sifaka.utils.state import RuleState

        class MyRule(BaseModel):
            # Configuration
            name: str
            
            # State
            _state: RuleState = PrivateAttr(default_factory=RuleState)
            
            def warm_up(self) -> None:
                """Initialize the rule."""
                if not self._state.initialized:
                    self._state.validator = self._create_validator()
                    self._state.initialized = True
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
        from sifaka.utils.state import CriticState

        class MyCritic(BaseModel):
            # Configuration
            name: str
            
            # State
            _state: CriticState = PrivateAttr(default_factory=CriticState)
            
            def warm_up(self) -> None:
                """Initialize the critic."""
                if not self._state.initialized:
                    self._state.model = self._load_model()
                    self._state.initialized = True
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
        from sifaka.utils.state import ModelState

        class MyModelProvider(BaseModel):
            # Configuration
            name: str
            
            # State
            _state: ModelState = PrivateAttr(default_factory=ModelState)
            
            def warm_up(self) -> None:
                """Initialize the model provider."""
                if not self._state.initialized:
                    self._state.client = self._create_client()
                    self._state.initialized = True
        ```
    """

    client: Optional[Any] = None
    token_counter: Optional[Any] = None
    tracer: Optional[Any] = None
    cache: Dict[str, Any] = {}


def create_state_manager(
    state_class: Type[T], **kwargs: Any
) -> StateManager[T]:
    """
    Create a state manager for a specific state class.

    Args:
        state_class: The state class to use
        **kwargs: Additional arguments to pass to the state class constructor

    Returns:
        A state manager for the specified state class
    """
    return StateManager(initializer=lambda: state_class(**kwargs))


def create_classifier_state(**kwargs: Any) -> StateManager[ClassifierState]:
    """
    Create a state manager for a classifier.

    Args:
        **kwargs: Additional arguments to pass to the ClassifierState constructor

    Returns:
        A state manager for a classifier
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
