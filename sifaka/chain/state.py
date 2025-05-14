"""
Chain State Management Module

This module provides state management utilities for the chain component.
It defines the chain state structure and provides functions for creating
and managing chain state.

## Overview
State management is a critical aspect of the chain component, as it needs
to track various resources, configurations, and execution statistics.
This module centralizes chain state management to ensure consistency and
maintainability.

## Components
1. **ChainState**: State structure for chain components
2. **create_chain_state**: Factory function for creating chain state managers
3. **ChainStateManager**: Specialized state manager for chain components

## Usage Examples
```python
from sifaka.chain.state import create_chain_state
from sifaka.models import OpenAIProvider
from sifaka.chain.validators import ValidationManager

# Create chain state
state_manager = create_chain_state()

# Initialize state
state_manager.update("model", OpenAIProvider("gpt-3.5-turbo") if state_manager else "")
state_manager.update("validation_manager", ValidationManager() if state_manager else "")
state_manager.update("initialized", True) if state_manager else ""
state_manager.update("execution_count", 0) if state_manager else ""
state_manager.update("result_cache", {}) if state_manager else ""

# Set metadata
state_manager.set_metadata("component_type", "chain") if state_manager else ""
state_manager.set_metadata("creation_time", time.time() if time else "" if state_manager else "")

# Access state
model = state_manager.get("model") if state_manager else ""
is_initialized = state_manager.get("initialized", False) if state_manager else ""
execution_count = state_manager.get("execution_count", 0) if state_manager else ""

# Update state
state_manager.update("execution_count", execution_count + 1) if state_manager else ""

# Rollback state if needed
state_manager.rollback() if state_manager else ""
```

## Error Handling
The state management utilities handle errors gracefully:
- Invalid state updates are validated through Pydantic
- State rollback is available for error recovery
- State reset is available for complete state reset

## Configuration
The state management utilities work with the following components:
- ChainState from utils.state module
- StateManager from utils.state module
"""

import time
from typing import Any, Dict, List, Optional

from ..utils.state import StateManager, create_chain_state as create_base_chain_state


class ChainStateManager(StateManager):
    """
    Specialized state manager for chain components.

    This class extends the base StateManager with chain-specific functionality,
    providing a standardized way to manage chain state with specialized methods
    for common chain operations.

    ## Architecture
    The ChainStateManager extends the base StateManager with chain-specific
    methods for tracking execution statistics, managing result caches, and
    handling chain-specific state operations.

    ## Lifecycle
    1. Initialization: Creates an empty chain state
    2. Operation: Updates state and tracks history
    3. Rollback: Reverts to previous state when needed
    4. Reset: Clears all state and history

    ## Error Handling
    The ChainStateManager provides rollback capabilities for error recovery
    and specialized methods for tracking execution errors.

    ## Examples
    ```python
    # Create a chain state manager
    manager = ChainStateManager()

    # Initialize chain state
    manager.initialize_chain(
        model=model,
        validators=validators,
        improver=improver,
        formatter=formatter,
        config=config
    ) if manager else ""

    # Track execution
    manager.track_execution_start() if manager else ""
    # ... run chain ...
    manager.track_execution_end(success=True, execution_time=0.5) if manager else ""

    # Access chain state
    model = manager.get_model() if manager else ""
    validators = manager.get_validators() if manager else ""
    execution_count = manager.get_execution_count() if manager else ""
    ```
    """

    def initialize_chain(
        self,
        model: Any,
        validators: List[Any],
        improver: Optional[Any] = None,
        formatter: Optional[Any] = None,
        config: Optional[Any] = None,
        name: str = "chain",
        description: str = "Sifaka chain for text generation and validation",
    ) -> None:
        """
        Initialize chain state.

        This method initializes the chain state with the provided components
        and configuration, setting up the initial state for chain execution.

        Args:
            model: The model to use for generation
            validators: The validators to use for validation
            improver: Optional improver for output improvement
            formatter: Optional formatter for result formatting
            config: Chain configuration
            name: Chain name
            description: Chain description

        Raises:
            ValueError: If any component is invalid
        """
        # Initialize state
        self.update("name", name)
        self.update("description", description)
        self.update("model", model)
        self.update("validators", validators)
        self.update("improver", improver)
        self.update("formatter", formatter)
        self.update("config", config)
        self.update("initialized", True)
        self.update("execution_count", 0)
        self.update("result_cache", {})

        # Set metadata
        self.set_metadata("component_type", "chain")
        self.set_metadata("creation_time", time.time())

    def track_execution_start(self) -> None:
        """
        Track the start of chain execution.

        This method updates the state to track the start of chain execution,
        incrementing the execution count and recording the start time.
        """
        # Track execution count
        execution_count = self.get("execution_count", 0)
        self.update("execution_count", execution_count + 1)

        # Record start time
        start_time = time.time()
        self.set_metadata("execution_start_time", start_time)

    def track_execution_end(
        self,
        success: bool,
        execution_time: float,
        error: Optional[Exception] = None,
    ) -> None:
        """
        Track the end of chain execution.

        This method updates the state to track the end of chain execution,
        recording the execution time, success status, and any errors.

        Args:
            success: Whether execution was successful
            execution_time: Execution time in seconds
            error: Optional error that occurred
        """
        # Record end time
        end_time = time.time()
        self.set_metadata("execution_end_time", end_time)

        # Update statistics
        self.set_metadata("last_execution_time", execution_time)
        self.set_metadata("last_execution_success", success)

        if success:
            success_count = self.get_metadata("success_count", 0)
            self.set_metadata("success_count", success_count + 1)
        else:
            failure_count = self.get_metadata("failure_count", 0)
            self.set_metadata("failure_count", failure_count + 1)

        if error:
            self.set_metadata("last_error", str(error))
            errors = self.get_metadata("errors", [])
            errors.append(str(error))
            self.set_metadata("errors", errors)

        # Update max execution time
        max_time = self.get_metadata("max_execution_time", 0)
        if execution_time > max_time:
            self.set_metadata("max_execution_time", execution_time)

    def get_model(self) -> Any:
        """
        Get the model.

        Returns:
            The model used for generation
        """
        return self.get("model")

    def get_validators(self) -> List[Any]:
        """
        Get the validators.

        Returns:
            The validators used for validation
        """
        validators = self.get("validators", [])
        if validators is None:
            return []
        return list(validators) if validators else []

    def get_improver(self) -> Optional[Any]:
        """
        Get the improver.

        Returns:
            The improver used for output improvement, or None if not set
        """
        return self.get("improver")

    def get_formatter(self) -> Optional[Any]:
        """
        Get the formatter.

        Returns:
            The formatter used for result formatting, or None if not set
        """
        return self.get("formatter")

    def get_execution_count(self) -> int:
        """
        Get the execution count.

        Returns:
            The number of times the chain has been executed
        """
        count = self.get("execution_count", 0)
        if count is None:
            return 0
        return int(count) if isinstance(count, (int, float, str)) else 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary of execution statistics
        """
        return {
            "execution_count": self.get("execution_count", 0),
            "success_count": self.get_metadata("success_count", 0),
            "failure_count": self.get_metadata("failure_count", 0),
            "last_execution_time": self.get_metadata("last_execution_time", 0),
            "max_execution_time": self.get_metadata("max_execution_time", 0),
            "last_execution_success": self.get_metadata("last_execution_success", True),
            "last_error": self.get_metadata("last_error", None),
            "cache_size": len(self.get("result_cache", {})),
        }


def create_chain_state(**kwargs: Any) -> ChainStateManager:
    """
    Create a state manager for a chain.

    This function creates a specialized ChainStateManager for managing chain state,
    initializing it with the provided keyword arguments.

    Args:
        **kwargs: Initial state values

    Returns:
        ChainStateManager: Specialized state manager for chain components
    """
    # Create base state manager
    state_manager = create_base_chain_state(**kwargs)

    # Create specialized state manager
    chain_state_manager = ChainStateManager()

    # Copy state from base state manager
    chain_state_manager._state = state_manager._state
    chain_state_manager._history = state_manager._history

    return chain_state_manager
