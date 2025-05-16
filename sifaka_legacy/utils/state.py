"""
State Management Module

A comprehensive module providing standardized state management utilities for all Sifaka components.

## Overview
This module implements a unified approach to state management across the Sifaka framework.
It provides classes and utilities for tracking, updating, and managing component state
in a consistent and type-safe manner.

## Components
1. **State**: Immutable state container for storing component state data
2. **StateManager**: Core utility class for managing component state with history tracking
3. **ComponentState**: Base class for specialized component state types
4. **Specialized State Classes**: Type-specific state containers for different component types
   - **ClassifierState**: State for classifier components
   - **RuleState**: State for rule and validator components
   - **CriticState**: State for critic components
   - **ModelState**: State for model provider components
   - **ChainState**: State for chain components
   - **AdapterState**: State for adapter components
5. **Factory Functions**: Utility functions for creating state managers for specific component types

## Usage Examples
```python
# Example 1: Basic state management
from sifaka.utils.state import StateManager
from pydantic import BaseModel, PrivateAttr

class MyComponent(BaseModel):
    name: str
    description: str = "A component with standardized state management"

    # State management (mutable)
    _state_manager = PrivateAttr(default_factory=StateManager)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Initialize state
        self._state_manager.update("initialized", False
        self._state_manager.update("cache", {}

    def process(self, input_data: Any) -> None:
        # Access and update state
        cache = self._state_manager.get("cache", {}
        if input_data in cache:
            return cache[input_data]

        # Process and update cache
        result = self._process_input(input_data) if self else ""
        cache[input_data] = result
        self._state_manager.update("cache", cache
        return result

# Example 2: Using specialized state managers
from sifaka.utils.state import create_classifier_state

class MyClassifier(BaseModel):
    name: str
    threshold: float = 0.5

    # Use specialized state manager
    _state_manager = PrivateAttr(default_factory=create_classifier_state)

    def warm_up(self) -> None:
        if not self._state_manager.get("initialized":
            # Initialize classifier-specific state
            self._state_manager.update("model", self._load_model() if self else ""
            self._state_manager.update("vectorizer", self._load_vectorizer() if self else ""
            self._state_manager.update("initialized", True
```

## Error Handling
The state management utilities handle errors by:
1. Providing immutable state objects to prevent accidental state corruption
2. Supporting state history and rollback for error recovery
3. Including error tracking in component state

## Configuration
State management can be configured through:
1. Using specialized state classes for different component types
2. Customizing state initialization with factory function parameters
3. Setting component-specific metadata through the state manager
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic, Type
from pydantic import BaseModel, Field, ConfigDict

T = TypeVar("T", bound=BaseModel)


class State(BaseModel):
    """
    Immutable state container for storing component state data.

    This class provides an immutable container for storing component state data
    and metadata. It uses Pydantic's frozen model feature to ensure immutability.

    ## Architecture
    The State class is designed as an immutable Pydantic model with two main
    dictionaries: data for state values and metadata for component information.

    ## Lifecycle
    States are created by the StateManager and should not be modified directly.
    New states are created through Pydantic's model_copy method when updates
    are needed.

    Attributes:
        data (Dict[str, Any]): Dictionary containing state values
        metadata (Dict[str, Any]): Dictionary containing metadata about the component
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StateManager:
    """
    Unified state management for all components.

    This class provides a standardized way to manage component state with history
    tracking and rollback capabilities. It serves as the primary interface for
    components to interact with their state.

    ## Architecture
    The StateManager maintains an immutable State object and a history list for
    tracking state changes. All state updates create a new State object rather
    than modifying the existing one.

    ## Lifecycle
    1. Initialization: Creates an empty state
    2. Operation: Updates state and tracks history
    3. Rollback: Reverts to previous state when needed
    4. Reset: Clears all state and history

    ## Error Handling
    The StateManager provides rollback capabilities for error recovery.

    ## Examples
    ```python
    # Create a state manager
    manager = StateManager()

    # Update state
    manager.update("initialized", True) if manager else ""
    manager.update("cache", {}) if manager else ""

    # Get state
    is_initialized = manager.get("initialized", False) if manager else ""
    cache = manager.get("cache", {}) if manager else ""

    # Set metadata
    manager.set_metadata("component_type", "Classifier") if manager else ""

    # Rollback to previous state
    manager.rollback() if manager else ""
    ```
    """

    def __init__(self) -> None:
        """Initialize a new state manager with empty state and history."""
        self._state: State = State()
        self._history: List[State] = []

    def update(self, key: str, value: Any) -> None:
        """
        Update state with history tracking.

        This method updates the state by creating a new State object with the
        updated value while preserving the previous state in history.

        Args:
            key (str): The state key to update
            value (Any): The new value to set
        """
        self._history.append(self._state)
        self._state = self._state.model_copy(update={"data": {**self._state.data, key: value}})

    def rollback(self) -> None:
        """
        Rollback to previous state.

        This method reverts the state to the previous version in history.
        If there is no history, the state remains unchanged.
        """
        if self._history:
            self._state = self._history.pop()

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get state value.

        Args:
            key (str): The state key to retrieve
            default (Any, optional): Default value if key doesn't exist. Defaults to None.

        Returns:
            Any: The state value or default if key doesn't exist
        """
        return self._state.data.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata value.

        This method updates the metadata by creating a new State object with the
        updated metadata value.

        Args:
            key (str): The metadata key to update
            value (Any): The new value to set
        """
        self._state = self._state.model_copy(
            update={"metadata": {**self._state.metadata, key: value}}
        )

    def get_metadata(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get metadata value.

        Args:
            key (str): The metadata key to retrieve
            default (Any, optional): Default value if key doesn't exist. Defaults to None.

        Returns:
            Any: The metadata value or default if key doesn't exist
        """
        return self._state.metadata.get(key, default)

    def reset(self) -> None:
        """
        Reset state to initial values.

        This method clears all state data and history, returning the state
        manager to its initial empty state.
        """
        self._state = State()
        self._history.clear()


class ComponentState(BaseModel):
    """
    Base class for component state.

    This class serves as the base for all specialized component state classes.
    It defines common state attributes that all components should track.

    ## Architecture
    ComponentState is designed as an immutable Pydantic model with common
    state attributes. Specialized component state classes inherit from this
    base class and add component-specific attributes.

    ## Lifecycle
    ComponentState objects are typically created by factory functions and
    used to initialize a StateManager. They should not be modified directly.

    Attributes:
        initialized (bool): Whether the component has been initialized
        error (Optional[str]): Error message if the component encountered an error
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)
    initialized: bool = False
    error: Optional[str] = None


class ClassifierState(ComponentState):
    """
    State for classifier components.

    This class defines the state attributes specific to classifier components,
    including machine learning models, vectorizers, and feature information.

    ## Architecture
    ClassifierState extends ComponentState with classifier-specific attributes
    for tracking ML models, vectorizers, pipelines, and feature information.

    ## Lifecycle
    ClassifierState objects are typically created by the create_classifier_state
    factory function and used to initialize a StateManager for classifier components.

    Attributes:
        model (Optional[Any]): The classifier model instance
        vectorizer (Optional[Any]): The text vectorizer instance
        pipeline (Optional[Any]): The processing pipeline instance
        feature_names (Dict[str, Any]): Dictionary of feature names and metadata
        cache (Dict[str, Any]): Cache for classification results
        dependencies_loaded (bool): Whether all dependencies have been loaded
    """

    model: Optional[Any] = None
    vectorizer: Optional[Any] = None
    pipeline: Optional[Any] = None
    feature_names: Dict[str, Any] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)
    dependencies_loaded: bool = False


class RuleState(ComponentState):
    """State for rules."""

    validator: Optional[Any] = None
    handler: Optional[Any] = None
    cache: Dict[str, Any] = Field(default_factory=dict)
    compiled_patterns: Dict[str, Any] = Field(default_factory=dict)


class CriticState(ComponentState):
    """State for critics."""

    model: Optional[Any] = None
    prompt_manager: Optional[Any] = None
    response_parser: Optional[Any] = None
    memory_manager: Optional[Any] = None
    cache: Dict[str, Any] = Field(default_factory=dict)


class ModelState(ComponentState):
    """State for model providers."""

    client: Optional[Any] = None
    token_counter: Optional[Any] = None
    tracer: Optional[Any] = None
    cache: Dict[str, Any] = Field(default_factory=dict)


class ChainState(ComponentState):
    """State for chains."""

    model: Optional[Any] = None
    generator: Optional[Any] = None
    validation_manager: Optional[Any] = None
    prompt_manager: Optional[Any] = None
    retry_strategy: Optional[Any] = None
    result_formatter: Optional[Any] = None
    critic: Optional[Any] = None
    cache: Dict[str, Any] = Field(default_factory=dict)


class AdapterState(ComponentState):
    """State for adapters."""

    adaptee: Optional[Any] = None
    adaptee_cache: Dict[str, Any] = Field(default_factory=dict)
    config_cache: Dict[str, Any] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)


class RetrieverState(ComponentState):
    """
    State for retriever components.

    This class defines the state attributes specific to retriever components,
    including query processors, result caches, and retrieval statistics.

    ## Architecture
    RetrieverState extends ComponentState with retriever-specific attributes
    for tracking query processing, caching, and performance metrics.

    ## Lifecycle
    RetrieverState objects are typically created by the create_retriever_state
    factory function and used to initialize a StateManager for retriever components.

    Attributes:
        query_processor (Optional[Any]): The query processor instance
        result_cache (Dict[str, Any]): Cache for retrieval results
        last_query (Optional[str]): The most recent query processed
        last_result (Optional[Any]): The most recent retrieval result
        execution_count (int): Number of retrieval operations performed
    """

    query_processor: Optional[Any] = None
    result_cache: Dict[str, Any] = Field(default_factory=dict)
    last_query: Optional[str] = None
    last_result: Optional[Any] = None
    execution_count: int = 0


class ModelProviderState(ComponentState):
    """
    State for model provider components.

    This class defines the state attributes specific to model provider components,
    including client information, token counters, and provider statistics.

    ## Architecture
    ModelProviderState extends ComponentState with provider-specific attributes
    for tracking API clients, token counters, and generation statistics.

    ## Lifecycle
    ModelProviderState objects are typically created by the create_model_provider_state
    factory function and used to initialize a StateManager for model provider components.

    Attributes:
        model_name (str): The name of the model being used
        config (Optional[Any]): The model configuration
        api_client (Optional[Any]): The API client instance
        token_counter (Optional[Any]): The token counter instance
        tracer (Optional[Any]): The tracer instance
        cache (Dict[str, Any]): Cache for generation results
    """

    model_name: str = ""
    config: Optional[Any] = None
    api_client: Optional[Any] = None
    token_counter: Optional[Any] = None
    tracer: Optional[Any] = None
    cache: Dict[str, Any] = Field(default_factory=dict)


class ManagerState(ComponentState):
    """
    State for manager components.

    This class defines the state attributes specific to manager components,
    such as prompt managers, response managers, and memory managers.

    ## Architecture
    ManagerState extends ComponentState with manager-specific attributes
    for tracking managed resources, caches, and operation statistics.

    ## Lifecycle
    ManagerState objects are typically created by manager-specific factory
    functions and used to initialize a StateManager for manager components.

    Attributes:
        resources (List[Any]): Managed resources
        cache (Dict[str, Any]): Cache for manager operations
    """

    resources: List[Any] = Field(default_factory=list)
    cache: Dict[str, Any] = Field(default_factory=dict)


def create_state_manager(state_class: Type[T], **kwargs: Any) -> StateManager:
    """
    Create a state manager for a specific state class.

    This function creates a StateManager initialized with the specified state class
    and optional initial values.

    Args:
        state_class (type[T]): The state class to use for initialization
        **kwargs (Any): Initial values for state attributes

    Returns:
        StateManager: A new state manager initialized with the specified state

    Example:
        ```python
        # Create a state manager with RuleState
        manager = create_state_manager(RuleState, initialized=True, validator=my_validator)
        ```
    """
    # VALID_DIRECT_INSTANTIATION: This is the base factory function that all other factory
    # functions build upon. Direct instantiation is acceptable here as this is the root
    # of the factory function hierarchy.
    manager = StateManager()
    state = state_class(**kwargs)
    for key, value in state.model_dump().items():
        manager.update(key, value)
    return manager


def create_classifier_state(**kwargs: Any) -> StateManager:
    """
    Create a state manager for a classifier component.

    This function creates a StateManager initialized with ClassifierState
    and optional initial values.

    Args:
        **kwargs (Any): Initial values for classifier state attributes

    Returns:
        StateManager: A new state manager initialized with ClassifierState

    Example:
        ```python
        # Create a classifier state manager
        manager = create_classifier_state(
            model=my_model,
            vectorizer=my_vectorizer,
            initialized=True
        )
        ```
    """
    return create_state_manager(ClassifierState, **kwargs)


def create_classifier_engine_state(**kwargs: Any) -> StateManager:
    """
    Create a state manager for a classifier engine component.

    This function creates a StateManager specialized for classifier engines
    with appropriate defaults.

    Args:
        **kwargs (Any): Initial values for classifier engine state attributes

    Returns:
        StateManager: A new state manager for classifier engines

    Example:
        ```python
        # Create a classifier engine state manager
        manager = create_classifier_engine_state(
            initialized=True,
            cache={}
        )
        ```
    """
    return create_manager_state(**kwargs)


def create_rule_state(**kwargs: Any) -> StateManager:
    """Create a state manager for a rule."""
    return create_state_manager(RuleState, **kwargs)


def create_critic_state(**kwargs: Any) -> StateManager:
    """Create a state manager for a critic."""
    return create_state_manager(CriticState, **kwargs)


def create_model_state(**kwargs: Any) -> StateManager:
    """Create a state manager for a model provider."""
    return create_state_manager(ModelState, **kwargs)


def create_chain_state(**kwargs: Any) -> StateManager:
    """Create a state manager for a chain."""
    return create_state_manager(ChainState, **kwargs)


def create_adapter_state(**kwargs: Any) -> StateManager:
    """Create a state manager for an adapter."""
    return create_state_manager(AdapterState, **kwargs)


def create_retriever_state(**kwargs: Any) -> StateManager:
    """
    Create a state manager for a retriever component.

    This function creates a StateManager initialized with RetrieverState
    and optional initial values.

    Args:
        **kwargs (Any): Initial values for retriever state attributes

    Returns:
        StateManager: A new state manager initialized with RetrieverState

    Example:
        ```python
        # Create a retriever state manager
        manager = create_retriever_state(
            query_processor=my_processor,
            initialized=True
        )
        ```
    """
    return create_state_manager(RetrieverState, **kwargs)


def create_manager_state(**kwargs: Any) -> StateManager:
    """
    Create a state manager for a manager component.

    This function creates a StateManager initialized with ManagerState
    and optional initial values.

    Args:
        **kwargs (Any): Initial values for manager state attributes

    Returns:
        StateManager: A new state manager initialized with ManagerState

    Example:
        ```python
        # Create a manager state manager
        manager = create_manager_state(
            resources=[prompt1, prompt2],
            initialized=True
        )
        ```
    """
    return create_state_manager(ManagerState, **kwargs)


def create_engine_state(**kwargs: Any) -> StateManager:
    """
    Create a state manager for an engine component.

    This function creates a StateManager specialized for engine components
    with appropriate defaults.

    Args:
        **kwargs (Any): Initial values for engine state attributes

    Returns:
        StateManager: A new state manager for engine components

    Example:
        ```python
        # Create an engine state manager
        manager = create_engine_state(
            initialized=True,
            cache={}
        )
        ```
    """
    return create_manager_state(**kwargs)


def create_response_manager_state(**kwargs: Any) -> StateManager:
    """
    Create a state manager for a response manager component.

    This function creates a StateManager specialized for response managers
    with appropriate defaults.

    Args:
        **kwargs (Any): Initial values for response manager state attributes

    Returns:
        StateManager: A new state manager for response managers
    """
    return create_manager_state(**kwargs)


def create_prompt_manager_state(**kwargs: Any) -> StateManager:
    """
    Create a state manager for a prompt manager component.

    This function creates a StateManager specialized for prompt managers
    with appropriate defaults.

    Args:
        **kwargs (Any): Initial values for prompt manager state attributes

    Returns:
        StateManager: A new state manager for prompt managers
    """
    return create_manager_state(**kwargs)


def create_memory_manager_state(**kwargs: Any) -> StateManager:
    """
    Create a state manager for a memory manager component.

    This function creates a StateManager specialized for memory managers
    with appropriate defaults.

    Args:
        **kwargs (Any): Initial values for memory manager state attributes

    Returns:
        StateManager: A new state manager for memory managers
    """
    return create_manager_state(**kwargs)


def create_model_provider_state(**kwargs: Any) -> StateManager:
    """
    Create a state manager for a model provider component.

    This function creates a StateManager initialized with ModelProviderState
    and optional initial values.

    Args:
        **kwargs (Any): Initial values for model provider state attributes

    Returns:
        StateManager: A new state manager initialized with ModelProviderState

    Example:
        ```python
        # Create a model provider state manager
        manager = create_model_provider_state(
            model_name="gpt-4",
            config=my_config,
            initialized=True
        )
        ```
    """
    return create_state_manager(ModelProviderState, **kwargs)
