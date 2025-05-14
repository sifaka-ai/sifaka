"""
Model Adapter Module

This module provides the ModelAdapter class for adapting existing model providers
to the Model interface from the chain system.
"""

import asyncio
import time
from typing import Any, Optional, Union, cast
from sifaka.interfaces.chain.components import Model
from sifaka.utils.errors.component import ModelError
from sifaka.utils.errors.results import ErrorResult
from sifaka.utils.errors.safe_execution import safely_execute_component_operation
from sifaka.utils.state import create_adapter_state


class ModelAdapter(Model):
    """
    Adapter for existing model providers.

    This adapter implements the Model interface for existing model providers,
    using the standardized state management pattern.

    ## Architecture
    The ModelAdapter follows the adapter pattern to wrap existing model providers:
    - Implements the Model interface from chain.interfaces
    - Uses standardized state management with _state_manager
    - Delegates to the wrapped model provider
    - Handles different model provider interfaces (invoke, generate, run, process)
    - Provides consistent error handling and statistics tracking

    ## Lifecycle
    1. **Initialization**: Adapter is created with a model provider
    2. **State Setup**: State manager is initialized with adapter state
    3. **Operation**: Adapter delegates to the model provider
    4. **Cleanup**: Resources are released when no longer needed

    ## Error Handling
    - ModelError: Raised when model generation fails
    - Tracks error statistics in state manager
    - Provides detailed error messages with component information

    Attributes:
        _model (Any): The wrapped model provider
        _name (str): The name of the adapter
        _description (str): The description of the adapter
        _state_manager (StateManager): The state manager for the adapter
    """

    def __init__(
        self, model: Any, name: Optional[str] = None, description: Optional[str] = None
    ) -> None:
        """
        Initialize the model adapter.

        Args:
            model: The model provider to adapt
            name: Optional name for the adapter
            description: Optional description for the adapter
        """
        self._model = model
        self._name = name or f"{type(model).__name__}Adapter"
        self._description = description or f"Adapter for {type(model).__name__}"
        self._state_manager = create_adapter_state()
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize adapter state."""
        self._state_manager.update("adaptee", self._model)
        self._state_manager.update("initialized", True)
        self._state_manager.update("cache", {})
        self._state_manager.set_metadata("component_type", "model_adapter")
        self._state_manager.set_metadata("adaptee_type", type(self._model).__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self._name

    @property
    def description(self) -> str:
        """Get adapter description."""
        return self._description

    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from

        Returns:
            The generated text

        Raises:
            ModelError: If text generation fails
        """
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        start_time = time.time()
        try:

            def generate_operation() -> Any:
                if hasattr(self._model, "invoke"):
                    return self._model.invoke(prompt)
                elif hasattr(self._model, "generate"):
                    return self._model.generate(prompt)
                elif hasattr(self._model, "run"):
                    return self._model.run(prompt)
                elif hasattr(self._model, "process"):
                    return self._model.process(prompt)
                else:
                    raise ModelError(f"Unsupported model provider: {type(self._model).__name__}")

            result = safely_execute_component_operation(
                operation=generate_operation,
                component_name=self.name,
                component_type="Model",
                error_class=ModelError,
            )

            # Handle the case where result might be an ErrorResult
            if isinstance(result, ErrorResult):
                raise ModelError(f"Model generation failed: {result.error_message}")

            # Convert result to string if it's not already
            generated_text = str(result)

            end_time = time.time()
            execution_time = end_time - start_time
            generation_count = self._state_manager.get_metadata("generation_count", 0)
            self._state_manager.set_metadata("generation_count", generation_count + 1)
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = (avg_time * generation_count + execution_time) / (generation_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)
            max_time = self._state_manager.get_metadata("max_execution_time", 0)

            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            return generated_text

        except Exception as e:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            if isinstance(e, ModelError):
                raise e
            raise ModelError(f"Model generation failed: {str(e)}")

    async def generate_async(self, prompt: str) -> str:
        """
        Generate text asynchronously.

        Args:
            prompt: The prompt to generate text from

        Returns:
            The generated text

        Raises:
            ModelError: If text generation fails
        """
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        start_time = time.time()
        try:
            generated_text: str

            if hasattr(self._model, "invoke_async"):
                result = await self._model.invoke_async(prompt)
                if isinstance(result, ErrorResult):
                    raise ModelError(f"Async model generation failed: {result.error_message}")
                generated_text = str(result)
            elif hasattr(self._model, "generate_async"):
                result = await self._model.generate_async(prompt)
                if isinstance(result, ErrorResult):
                    raise ModelError(f"Async model generation failed: {result.error_message}")
                generated_text = str(result)
            elif hasattr(self._model, "run_async"):
                result = await self._model.run_async(prompt)
                if isinstance(result, ErrorResult):
                    raise ModelError(f"Async model generation failed: {result.error_message}")
                generated_text = str(result)
            elif hasattr(self._model, "process_async"):
                result = await self._model.process_async(prompt)
                if isinstance(result, ErrorResult):
                    raise ModelError(f"Async model generation failed: {result.error_message}")
                generated_text = str(result)
            else:
                loop = asyncio.get_event_loop()
                generated_text = await loop.run_in_executor(None, self.generate, prompt)
                return generated_text

            end_time = time.time()
            execution_time = end_time - start_time
            generation_count = self._state_manager.get_metadata("generation_count", 0)
            self._state_manager.set_metadata("generation_count", generation_count + 1)
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = (avg_time * generation_count + execution_time) / (generation_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)
            max_time = self._state_manager.get_metadata("max_execution_time", 0)

            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            return generated_text

        except Exception as e:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            if isinstance(e, ModelError):
                raise e
            raise ModelError(f"Async model generation failed: {str(e)}")
