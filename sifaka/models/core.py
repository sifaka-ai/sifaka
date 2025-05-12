"""
Core Model Provider Implementation

This module provides the ModelProviderCore class which is the main interface
for model providers, delegating to specialized components for better separation of concerns.

## Overview
The ModelProviderCore class implements a component-based architecture for model providers,
delegating functionality to specialized managers and services. It serves as the foundation
for all model provider implementations in Sifaka, providing a consistent interface while
allowing for provider-specific customization.

## Components
- **ModelProviderCore**: Main provider class that delegates to specialized components
- **ClientManager**: Manages API client creation and lifecycle
- **TokenCounterManager**: Manages token counting functionality
- **TracingManager**: Manages tracing and logging
- **GenerationService**: Handles text generation and error handling

## Usage Examples
```python
# Create a custom provider that extends ModelProviderCore
class MyProvider(ModelProviderCore):
    def _create_default_client(self) -> APIClient:
        return MyAPIClient(api_key=self.config.api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        return MyTokenCounter(model=self.model_name)

# Use the provider
provider = MyProvider(model_name="my-model")
response = provider.generate("Hello, world!")
```

## Error Handling
The module implements standardized error handling patterns:
- Input validation with clear error messages
- Structured error recording and propagation
- Consistent error types for different failure modes
- Detailed error metadata for debugging

## Configuration
The module uses the standardized configuration approach from utils/config.py,
with provider-specific extensions as needed.
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar, TYPE_CHECKING

from pydantic import PrivateAttr

# Import interfaces directly to avoid circular dependencies
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter
from sifaka.interfaces.model import ModelProviderProtocol

# Import base classes using string type annotations for forward references
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sifaka.models.base import ModelProvider

from sifaka.utils.config.models import ModelConfig
from sifaka.models.managers.client import ClientManager
from sifaka.models.managers.token_counter import TokenCounterManager
from sifaka.models.managers.tracing import TracingManager
from sifaka.models.services.generation import GenerationService
from sifaka.utils.tracing import Tracer
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_model_state

logger = get_logger(__name__)

# Type variables for generic model provider types
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

    ## Error Handling
    The class implements standardized error handling patterns:
    - Input validation with clear error messages
    - Structured error recording and propagation
    - Consistent error types for different failure modes
    - Detailed error metadata for debugging

    ## State Management
    The class uses a standardized state management approach:
    - Single _state_manager attribute for all mutable state
    - State initialization during construction
    - State access through state manager
    - Clear separation of configuration and state
    - State components:
      - model_name: Name of the language model
      - config: Model configuration
      - client_manager: API client manager
      - token_counter_manager: Token counter manager
      - tracing_manager: Tracing manager
      - generation_service: Generation service
      - initialized: Initialization status
      - cache: Temporary data storage

    Type Parameters:
        C: The configuration type, must be a subclass of ModelConfig

    Examples:
        ```python
        # Create a custom provider that extends ModelProviderCore
        class MyProvider(ModelProviderCore):
            def _create_default_client(self) -> APIClient:
                return MyAPIClient(api_key=self.config.api_key)

            def _create_default_token_counter(self) -> TokenCounter:
                return MyTokenCounter(model=self.model_name)

        # Use the provider
        provider = MyProvider(model_name="my-model")
        response = provider.generate("Hello, world!")

        # With custom configuration
        from sifaka.utils.config.models import ModelConfig
        config = ModelConfig(temperature=0.7, max_tokens=2000)
        provider = MyProvider(model_name="my-model", config=config)

        # Error handling
        try:
            response = provider.generate("Tell me about quantum computing")
        except ValueError as e:
            print(f"Input validation error: {e}")
        except RuntimeError as e:
            print(f"Generation error: {e}")
            # Implement fallback strategy
        ```
    """

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_model_state)

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

        Example:
            ```python
            # Basic initialization
            provider = ModelProviderCore(model_name="gpt-4")

            # With custom configuration
            from sifaka.utils.config.models import ModelConfig
            config = ModelConfig(temperature=0.7, max_tokens=2000)
            provider = ModelProviderCore(model_name="gpt-4", config=config)

            # With custom dependencies
            provider = ModelProviderCore(
                model_name="gpt-4",
                config=config,
                api_client=custom_client,
                token_counter=custom_counter,
                tracer=custom_tracer
            )
            ```
        """
        import time

        # Initialize state
        self._state_manager.update("model_name", model_name)
        self._state_manager.update("config", config or self._create_default_config())
        self._state_manager.update("initialized", False)
        self._state_manager.update("cache", {})

        # Store dependencies for later initialization
        self._state_manager.update("_api_client", api_client)
        self._state_manager.update("_token_counter", token_counter)
        self._state_manager.update("_tracer", tracer)

        # Set metadata
        self._state_manager.set_metadata("component_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            The name of the language model
        """
        return self._state_manager.get("model_name")

    @property
    def config(self) -> C:
        """
        Get the model configuration.

        Returns:
            The current model configuration
        """
        return self._state_manager.get("config")

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

        Example:
            ```python
            # Basic token counting
            provider = OpenAIProvider(model_name="gpt-4")
            count = provider.count_tokens("How many tokens is this?")
            print(f"Token count: {count}")

            # Using for optimization
            prompt = "This is a long prompt..."
            count = provider.count_tokens(prompt)
            if count > 1000:
                # Truncate or summarize if too long
                prompt = prompt[:500] + "..."
                print(f"Truncated prompt to reduce tokens")
            ```
        """
        # Ensure component is initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Process input
        start_time = __import__("time").time()

        # Define the operation
        def operation():
            # Actual processing logic
            result = self._count_tokens_impl(text)
            return result

        # Use standardized error handling
        from sifaka.utils.error_patterns import safely_execute_component_operation

        token_count = safely_execute_component_operation(
            operation=operation,
            component_name=self.name,
            component_type=self.__class__.__name__,
            additional_metadata={"input_type": "text", "method": "count_tokens"},
        )

        # Update statistics
        processing_time = __import__("time").time() - start_time
        self._update_token_count_statistics(token_count, processing_time_ms=processing_time * 1000)

        return token_count

    def _count_tokens_impl(self, text: str) -> int:
        """Implement token counting logic."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # Get token counter manager from state
        token_counter_manager = self._state_manager.get("token_counter_manager")
        token_count = token_counter_manager.count_tokens(text)

        # Get tracing manager from state
        tracing_manager = self._state_manager.get("tracing_manager")
        tracing_manager.trace_event(
            "token_count",
            {
                "text_length": len(text),
                "token_count": token_count,
            },
        )

        return token_count

    def _update_token_count_statistics(self, token_count: int, processing_time_ms: float) -> None:
        """Update statistics after token counting."""
        # Update statistics in state
        count_stats = self._state_manager.get("token_count_stats", {})
        count_stats["total_tokens_counted"] = (
            count_stats.get("total_tokens_counted", 0) + token_count
        )
        count_stats["count_operations"] = count_stats.get("count_operations", 0) + 1
        count_stats["total_processing_time_ms"] = (
            count_stats.get("total_processing_time_ms", 0) + processing_time_ms
        )
        self._state_manager.update("token_count_stats", count_stats)

    def generate(self, prompt: str, **kwargs) -> str:
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

        Example:
            ```python
            # Basic generation
            provider = OpenAIProvider(model_name="gpt-4")
            response = provider.generate("Explain quantum computing in simple terms.")

            # With parameter overrides
            response = provider.generate(
                "Write a creative story.",
                temperature=0.9,
                max_tokens=2000
            )

            # Error handling
            try:
                response = provider.generate("Explain quantum computing")
            except ValueError as e:
                print(f"Input error: {e}")
            except RuntimeError as e:
                print(f"Generation failed: {e}")
                # Implement fallback strategy
            ```
        """
        # Ensure component is initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Process input
        start_time = __import__("time").time()

        # Define the operation
        def operation():
            # Actual processing logic
            result = self._process_input(prompt, **kwargs)
            return result

        # Use standardized error handling
        from sifaka.utils.error_patterns import safely_execute_component_operation

        result = safely_execute_component_operation(
            operation=operation,
            component_name=self.name,
            component_type=self.__class__.__name__,
            additional_metadata={"input_type": "prompt", "method": "generate"},
        )

        # Update statistics
        processing_time = __import__("time").time() - start_time
        self._update_statistics(result, processing_time_ms=processing_time * 1000)

        return result

    def _process_input(self, prompt: str, **kwargs) -> str:
        """Process the input prompt and generate text."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Check cache if enabled
        cache = self._state_manager.get("cache", {})
        cache_key = f"{prompt}_{str(kwargs)}"
        if cache_key in cache:
            self._state_manager.set_metadata("cache_hit", True)
            return cache[cache_key]

        # Update config with any override kwargs
        config = ModelConfig(
            temperature=kwargs.pop("temperature", self.config.temperature),
            max_tokens=kwargs.pop("max_tokens", self.config.max_tokens),
            api_key=kwargs.pop("api_key", self.config.api_key),
            trace_enabled=kwargs.pop("trace_enabled", self.config.trace_enabled),
        )

        # Verify API key is set before attempting to generate
        if not config.api_key:
            model_specific_env = (
                f"{self.__class__.__name__.replace('Provider', '').upper()}_API_KEY"
            )
            raise ValueError(
                f"API key is missing. Please provide an API key either by setting the "
                f"{model_specific_env} environment variable or by passing it explicitly "
                f"via the api_key parameter or config."
            )

        # Get generation service from state
        generation_service = self._state_manager.get("generation_service")

        # Generate text
        result = generation_service.generate(prompt, config)

        # Update cache
        if len(cache) < 100:  # Limit cache size
            cache[cache_key] = result
            self._state_manager.update("cache", cache)

        return result

    def _update_statistics(self, result: str, processing_time_ms: float) -> None:
        """
        Update statistics after generation.

        This method updates the generation statistics in the state manager,
        tracking metrics like generation count and total processing time.
        These statistics can be used for monitoring and optimization.

        Args:
            result: The generated text (used for potential future metrics like token count)
            processing_time_ms: The processing time in milliseconds
        """
        # Track generation count
        generation_stats = self._state_manager.get("generation_stats", {})
        generation_stats["generation_count"] = generation_stats.get("generation_count", 0) + 1
        generation_stats["total_generation_time_ms"] = (
            generation_stats.get("total_generation_time_ms", 0) + processing_time_ms
        )

        # Could add metrics based on result in the future
        # For example: generation_stats["avg_token_count"] = len(result.split())

        self._state_manager.update("generation_stats", generation_stats)

    def _create_token_counter_manager(
        self, token_counter: Optional[TokenCounter]
    ) -> TokenCounterManager:
        """
        Create a token counter manager.

        This method creates a specialized TokenCounterManager instance
        that uses the provider's token counter creation method. It uses
        a closure to bind the provider's _create_default_token_counter
        method to the manager.

        Args:
            token_counter: Optional token counter to use

        Returns:
            A token counter manager configured for this provider

        Example:
            ```python
            # Internal usage
            token_counter = None  # No explicit counter provided
            manager = self._create_token_counter_manager(token_counter)

            # The manager will use the provider's token counter creation method
            counter = manager.get_token_counter()  # Uses _create_default_token_counter
            ```
        """

        class ConcreteTokenCounterManager(TokenCounterManager):
            def _create_default_token_counter(self2) -> TokenCounter:
                return self._create_default_token_counter()

        return ConcreteTokenCounterManager(self._state_manager.get("model_name"), token_counter)

    def _create_client_manager(self, api_client: Optional[APIClient]) -> ClientManager:
        """
        Create a client manager.

        This method creates a specialized ClientManager instance
        that uses the provider's client creation method. It uses
        a closure to bind the provider's _create_default_client
        method to the manager.

        Args:
            api_client: Optional API client to use

        Returns:
            A client manager configured for this provider

        Example:
            ```python
            # Internal usage
            api_client = None  # No explicit client provided
            manager = self._create_client_manager(api_client)

            # The manager will use the provider's client creation method
            client = manager.get_client()  # Uses _create_default_client
            ```
        """

        class ConcreteClientManager(ClientManager):
            def _create_default_client(self2) -> APIClient:
                return self._create_default_client()

        return ConcreteClientManager(
            self._state_manager.get("model_name"), self._state_manager.get("config"), api_client
        )

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

        Example:
            ```python
            # Implementation in a concrete subclass
            def _create_default_client(self) -> APIClient:
                from sifaka.models.clients.openai import OpenAIClient
                return OpenAIClient(
                    api_key=self.config.api_key,
                    model=self.model_name
                )
            ```
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

        Example:
            ```python
            # Implementation in a concrete subclass
            def _create_default_token_counter(self) -> TokenCounter:
                from sifaka.models.counters.openai import OpenAITokenCounter
                return OpenAITokenCounter(model=self.model_name)
            ```
        """
        ...

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

        Example:
            ```python
            # Explicit warm-up
            provider = OpenAIProvider(model_name="gpt-4")
            provider.warm_up()  # Initialize resources

            # Implicit warm-up (happens automatically)
            provider = OpenAIProvider(model_name="gpt-4")
            response = provider.generate("Hello")  # Calls warm_up internally
            ```
        """
        try:
            # Check if already initialized
            if self._state_manager.get("initialized", False):
                logger.debug(f"Component {self.name} already initialized")
                return

            # Initialize resources
            self._initialize_resources()

            # Mark as initialized
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("warm_up_time", __import__("time").time())

            logger.debug(f"Component {self.name} warmed up successfully")

        except Exception as e:
            self._record_error(e)
            from sifaka.utils.errors.base import InitializationError

            raise InitializationError(f"Failed to warm up component {self.name}: {str(e)}") from e

    def _initialize_resources(self) -> None:
        """Initialize all resources needed by the model provider."""
        # Create managers and store in state
        token_counter = self._state_manager.get("_token_counter")
        token_counter_manager = self._create_token_counter_manager(token_counter)

        api_client = self._state_manager.get("_api_client")
        client_manager = self._create_client_manager(api_client)

        tracer = self._state_manager.get("_tracer")
        tracing_manager = TracingManager(
            self._state_manager.get("model_name"), self._state_manager.get("config"), tracer
        )

        self._state_manager.update("token_counter_manager", token_counter_manager)
        self._state_manager.update("client_manager", client_manager)
        self._state_manager.update("tracing_manager", tracing_manager)

        # Create services and store in state
        generation_service = GenerationService(
            self._state_manager.get("model_name"),
            client_manager,
            token_counter_manager,
            tracing_manager,
        )
        self._state_manager.update("generation_service", generation_service)

    def cleanup(self) -> None:
        """
        Clean up component resources.

        This method releases all resources used by the model provider,
        including API clients, token counters, and tracing resources.
        It also clears the cache and resets the initialization flag.

        The method is designed to be safe to call multiple times and
        will not raise exceptions even if cleanup fails.

        Example:
            ```python
            # Create and use a provider
            provider = OpenAIProvider(model_name="gpt-4")
            response = provider.generate("Hello")

            # Clean up when done
            provider.cleanup()
            ```
        """
        try:
            # Release resources
            self._release_resources()

            # Clear cache
            self._state_manager.update("cache", {})

            # Reset initialization flag
            self._state_manager.update("initialized", False)

            logger.debug(f"Component {self.name} cleaned up successfully")

        except Exception as e:
            # Log but don't raise
            logger.error(f"Failed to clean up component {self.name}: {str(e)}")

    def _release_resources(self) -> None:
        """Release all resources used by the model provider."""
        # Release client resources if possible
        client_manager = self._state_manager.get("client_manager")
        if client_manager and hasattr(client_manager, "close"):
            client_manager.close()

        # Release token counter resources if possible
        token_counter_manager = self._state_manager.get("token_counter_manager")
        if token_counter_manager and hasattr(token_counter_manager, "close"):
            token_counter_manager.close()

        # Release tracing resources if possible
        tracing_manager = self._state_manager.get("tracing_manager")
        if tracing_manager and hasattr(tracing_manager, "close"):
            tracing_manager.close()

    def _record_error(self, error: Exception) -> None:
        """Record an error in the state manager."""
        from sifaka.utils.common import record_error

        record_error(
            error=error,
            component_name=self.name,
            component_type=self.__class__.__name__,
            state_manager=self._state_manager,
        )

    @property
    def name(self) -> str:
        """
        Get the provider name.

        This property returns a human-readable name for the provider,
        combining the class name and model name. It is used for logging,
        error messages, and debugging.

        Returns:
            The provider name as a string in the format "ClassName-model_name"

        Example:
            ```python
            provider = OpenAIProvider(model_name="gpt-4")
            print(provider.name)  # Outputs: "OpenAIProvider-gpt-4"
            ```
        """
        return f"{self.__class__.__name__}-{self._state_manager.get('model_name')}"
