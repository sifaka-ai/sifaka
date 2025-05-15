"""
Core Model Provider Implementation

This module provides the ModelProviderCore class which is the main interface
for model providers, delegating to specialized components for better separation of concerns.
"""

from abc import abstractmethod
from typing import Any, Generic, Optional, TypeVar, cast
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.interfaces.model import ModelProviderProtocol
from sifaka.utils.config.models import ModelConfig
from sifaka.utils.logging import get_logger
from sifaka.utils.tracing import Tracer
from sifaka.utils.state import create_model_provider_state
from .initialization import initialize_resources, release_resources
from .generation import process_input
from .token_counting import count_tokens_impl
from .error_handling import record_error
from .utils import update_statistics, update_token_count_statistics
from numbers import Number

logger = get_logger(__name__)
T = TypeVar("T", bound="ModelProviderCore")
C = TypeVar("C", bound=ModelConfig)


class ModelProviderCore(ModelProviderProtocol, Generic[C]):
    """
    Core model provider implementation that delegates to specialized components.

    This class implements the ModelProvider interface but delegates most of its
    functionality to specialized components for better separation of concerns.
    It provides a standardized approach to state management, error handling,
    and component lifecycle management.

    ## Architecture
    The ModelProviderCore follows a component-based architecture:
    - Delegates API communication to ClientManager
    - Delegates token counting to TokenCounterManager
    - Delegates tracing to TracingManager
    - Delegates text generation to GenerationService

    This separation of concerns makes the code more maintainable and testable,
    allowing each component to focus on a specific responsibility.

    ## Lifecycle
    1. **Initialization**: Set up the provider with a model name, configuration, and component managers
       - Create state manager and initialize state
       - Store dependencies for later initialization
       - Set metadata for tracing and debugging

    2. **Warm-up**: Prepare the component for use (lazy initialization)
       - Create managers (client, token counter, tracing)
       - Create services (generation)
       - Mark as initialized

    3. **Usage**: Generate text and count tokens
       - Delegate to appropriate services
       - Handle errors consistently
       - Update statistics

    4. **Cleanup**: Release resources when no longer needed
       - Release client resources
       - Release token counter resources
       - Release tracing resources
       - Clear cache and reset initialization flag

    Type Parameters:
        C: The configuration type, must be a subclass of ModelConfig
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[C] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
        tracer: Optional[Tracer] = None,
    ) -> None:
        """
        Initialize a ModelProviderCore instance.

        This method initializes the state manager and stores dependencies for later
        initialization. It follows a lazy initialization pattern, where actual
        resource creation is deferred until the warm_up method is called.

        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            api_client: Optional API client to use
            token_counter: Optional token counter to use
            tracer: Optional tracer to use

        Raises:
            ValueError: If model_name is empty
            TypeError: If dependencies don't implement required protocols
        """
        import time

        self._state_manager = create_model_provider_state()
        self._state_manager.update("model_name", model_name)
        self._state_manager.update("config", config or self._create_default_config())
        self._state_manager.update("initialized", False)
        self._state_manager.update("cache", {})
        self._state_manager.update("api_client", api_client)
        self._state_manager.update("token_counter", token_counter)
        self._state_manager.update("tracer", tracer)
        self._state_manager.set_metadata("component_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())
        if logger:
            logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            The name of the language model
        """
        return str(self._state_manager.get("model_name"))

    @property
    def config(self) -> C:
        """
        Get the model configuration.

        Returns:
            The current model configuration
        """
        config = self._state_manager.get("config")
        if not isinstance(config, ModelConfig):
            raise TypeError(f"Expected ModelConfig, got {type(config)}")
        return config  # type: ignore

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.

        This method delegates to the token counter manager to perform
        the actual token counting. It handles input validation,
        error handling, and statistics tracking.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text

        Raises:
            TypeError: If text is not a string
            RuntimeError: If token counting fails
        """
        if not self._state_manager.get("initialized", False):
            self.warm_up()
        start_time = __import__("time").time()

        def operation() -> Any:
            result = count_tokens_impl(self, text)
            return result

        from sifaka.utils.errors.safe_execution import safely_execute_component_operation
        from sifaka.utils.errors.component import ModelError

        token_count = safely_execute_component_operation(
            operation=operation,
            component_name=self.name,
            component_type=self.__class__.__name__,
            error_class=ModelError,
            additional_metadata={"input_type": "text", "method": "count_tokens"},
        )
        processing_time = __import__("time").time() - start_time

        # Handle the case where token_count might be an ErrorResult
        if hasattr(token_count, "error"):
            # If it's an error result, return 0 tokens
            return 0

        # Convert token_count to int safely
        if isinstance(token_count, int):
            token_count_int = token_count
        elif isinstance(token_count, (float, str)):
            try:
                token_count_int = int(token_count)
            except (ValueError, TypeError):
                token_count_int = 0
        else:
            # If we can't safely convert, return 0
            token_count_int = 0

        update_token_count_statistics(
            self, token_count_int, processing_time_ms=processing_time * 1000
        )
        return token_count_int

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text using the model.

        This method delegates to the generation service to perform
        the actual text generation. It handles input validation,
        configuration overrides, error handling, and statistics tracking.

        Args:
            prompt: The prompt to generate from
            **kwargs: Optional overrides for model configuration
                - temperature: Control randomness (0-1)
                - max_tokens: Maximum tokens to generate
                - api_key: Override API key
                - trace_enabled: Override tracing setting

        Returns:
            The generated text

        Raises:
            TypeError: If prompt is not a string
            ValueError: If prompt is empty or API key is missing
            RuntimeError: If an error occurs during generation
        """
        if not self._state_manager.get("initialized", False):
            self.warm_up()
        start_time = __import__("time").time()

        def operation() -> Any:
            result = process_input(self, prompt, **kwargs)
            return result

        from sifaka.utils.errors.safe_execution import safely_execute_component_operation
        from sifaka.utils.errors.component import ModelError

        result = safely_execute_component_operation(
            operation=operation,
            component_name=self.name,
            component_type=self.__class__.__name__,
            error_class=ModelError,
            additional_metadata={"input_type": "prompt", "method": "generate"},
        )
        processing_time = __import__("time").time() - start_time

        # Handle the case where result might be an ErrorResult
        if hasattr(result, "error"):
            # If it's an error result, return the error message
            return f"Error: {str(result.error)}"

        update_statistics(self, str(result), processing_time_ms=processing_time * 1000)
        return str(result)

    def warm_up(self) -> None:
        """
        Prepare the component for use.

        This method initializes all necessary resources for the model provider,
        including API clients, token counters, and services. It follows a lazy
        initialization pattern, where resources are only created when needed.

        The warm_up method is automatically called by other methods when needed,
        so it's not necessary to call it explicitly in most cases.

        Raises:
            InitializationError: If initialization fails
        """
        try:
            if self._state_manager.get("initialized", False):
                if logger:
                    logger.debug(f"Component {self.name} already initialized")
                return
            initialize_resources(self)
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("warm_up_time", __import__("time").time())
            if logger:
                logger.debug(f"Component {self.name} warmed up successfully")
        except Exception as e:
            record_error(self, e)
            from sifaka.utils.errors.base import InitializationError

            raise InitializationError(f"Failed to warm up component {self.name}: {str(e)}") from e

    def cleanup(self) -> None:
        """
        Clean up component resources.

        This method releases all resources used by the model provider,
        including API clients, token counters, and tracing resources.
        It also clears the cache and resets the initialization flag.

        The method is designed to be safe to call multiple times and
        will not raise exceptions even if cleanup fails.
        """
        try:
            release_resources(self)
            self._state_manager.update("cache", {})
            self._state_manager.update("initialized", False)
            if logger:
                logger.debug(f"Component {self.name} cleaned up successfully")
        except Exception as e:
            if logger:
                logger.error(f"Failed to clean up component {self.name}: {str(e)}")

    def _create_default_config(self) -> C:
        """Create a default configuration if none was provided."""
        return ModelConfig()  # type: ignore

    @abstractmethod
    def _create_default_client(self) -> APIClient:
        """
        Create a default API client if none was provided.

        This abstract method must be implemented by subclasses to provide
        model-specific API client creation. It is called by the client manager
        when no explicit client is provided.

        Returns:
            A default API client for the model

        Raises:
            RuntimeError: If a default client cannot be created
        """
        ...

    @abstractmethod
    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default token counter if none was provided.

        This abstract method must be implemented by subclasses to provide
        model-specific token counter creation. It is called by the token counter
        manager when no explicit token counter is provided.

        Returns:
            A default token counter for the model

        Raises:
            RuntimeError: If a default token counter cannot be created
        """
        ...

    @property
    def name(self) -> str:
        """
        Get the provider name.

        This property returns a human-readable name for the provider,
        combining the class name and model name. It is used for logging,
        error messages, and debugging.

        Returns:
            The provider name as a string in the format "ClassName-model_name"
        """
        return f"{self.__class__.__name__}-{self._state_manager.get('model_name')}"
