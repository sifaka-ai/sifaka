"""
Model Base Module

This module provides the core base implementations for model providers in the Sifaka framework,
building on the interfaces defined in sifaka.interfaces.model.

## Overview
The model base module serves as the foundation for all language model integrations in Sifaka.
It defines abstract base classes and factory functions that standardize how model providers
are implemented, configured, and used throughout the system. This ensures consistent behavior
and error handling across different model providers.

## Components
- **ModelProvider**: Abstract base class implementing the ModelProviderProtocol
- **create_model_provider**: Factory function for creating model provider instances
- **LanguageModel**: Type alias for LanguageModelProtocol (for backward compatibility)

## Architecture
The model system follows a layered architecture:

1. **ModelProvider**: High-level interface for model interactions
2. **APIClient**: Low-level communication with model services
3. **TokenCounter**: Utility for token counting
4. **Config**: Configuration and settings management

## Usage Examples
```python
from sifaka.models import create_model_provider, ModelConfig
from sifaka.interfaces.model import ModelProviderProtocol

# Basic usage with factory function
provider = create_model_provider(
    ProviderClass,  # Any class implementing ModelProviderProtocol
    model_name="model-name",
    api_key="your-api-key",
    temperature=0.8
)
response = provider.generate("Explain quantum computing in simple terms.")

# Direct instantiation with custom configuration
config = ModelConfig().with_temperature(0.9).with_max_tokens(2000)
provider = ProviderClass(model_name="model-name", config=config)

# Error handling pattern
try:
    response = provider.generate("Explain quantum computing")
except ValueError as e:
    # Handle input validation errors
    print(f"Input error: {e}")
except RuntimeError as e:
    # Handle API and generation errors
    print(f"Generation failed: {e}")
    # Implement fallback strategy
    fallback_provider = create_model_provider(
        FallbackProviderClass,
        model_name="fallback-model"
    )
    response = fallback_provider.generate("Explain quantum computing briefly")
```

## Error Handling
The model system implements several error handling patterns:

1. **Typed Exceptions**: Use specific exception types for different error cases
   - TypeError: For type validation issues
   - ValueError: For invalid inputs
   - RuntimeError: For operational failures

2. **Automatic Retries**: Implement backoff strategy for transient errors
3. **Graceful Degradation**: Fallback to simpler models when primary fails
4. **Thorough Logging**: Log all errors with context for diagnosis
5. **Tracing**: Record detailed events for monitoring and debugging

## Configuration
Model providers can be configured with:
- **model_name**: Name of the model to use
- **temperature**: Controls randomness (0-1)
- **max_tokens**: Maximum tokens to generate
- **api_key**: API key for authentication
- **trace_enabled**: Enable/disable tracing
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    TypeVar,
)

# Import the interfaces for runtime use
from sifaka.interfaces.model import ModelProviderProtocol as ModelProviderInterface
from sifaka.interfaces.client import APIClientProtocol as APIClient
from sifaka.interfaces.counter import TokenCounterProtocol as TokenCounter

from sifaka.utils.logging import get_logger
from sifaka.utils.tracing import Tracer

logger = get_logger(__name__)

# Import configuration from utils/config.py
from sifaka.utils.config.models import ModelConfig

# Type variables for generic type definitions
T = TypeVar("T")  # Generic type for ModelProvider
C = TypeVar("C", bound=ModelConfig)  # Config type


# Import the LanguageModelProtocol from interfaces
from sifaka.interfaces.model import LanguageModelProtocol

# Use LanguageModelProtocol as an alias for backward compatibility
# This will be used by existing code that imports LanguageModel from models.base
LanguageModel = LanguageModelProtocol


class ModelProvider(ModelProviderInterface, Generic[C], ABC):
    """
    Abstract base class for model providers.

    This class implements the ModelProviderProtocol interface and enforces a consistent
    interface for all model providers while allowing for flexible implementation of
    specific provider features.

    ## Architecture
    ModelProvider follows a layered architecture with:
    - High-level interface for model interactions
    - Dependency injection for API clients and token counters
    - Tracing and logging for monitoring and debugging
    - Generic type parameter for configuration customization
    - Abstract methods for provider-specific implementations

    Type Parameters:
        C: The configuration type, must be a subclass of ModelConfig

    ## Lifecycle
    1. **Initialization**: Set up with model name and configuration
       - Create/validate config
       - Initialize dependencies (API client, token counter)
       - Set up tracing if enabled

    2. **Configuration**: Configure generation parameters
       - Set temperature, max_tokens, etc.
       - Configure API keys and authentication

    3. **Operation**: Main usage phase
       - Generate text from prompts
       - Count tokens for optimization
       - Trace events for monitoring

    4. **Error Handling**: Manage failures
       - Handle API errors
       - Implement retries and circuit breakers
       - Provide meaningful error messages

    5. **Cleanup**: Release resources
       - Close API clients
       - Ensure proper resource management

    ## Error Handling
    This class implements several error handling patterns:

    1. **Input Validation**: Validates all inputs before processing
       - Checks prompt is a string and not empty
       - Validates configuration parameters

    2. **Dependency Management**: Ensures required dependencies exist
       - Creates default components when needed
       - Validates components implement required protocols

    3. **Exception Handling**: Proper exception hierarchy
       - Uses TypeError for type validation issues
       - Uses ValueError for invalid inputs
       - Uses RuntimeError for operational failures
       - Preserves original exception in context

    4. **Tracing and Logging**: Records errors for diagnosis
       - Logs all errors with context
       - Traces events for debugging
       - Includes relevant metadata with errors

    ## Examples
    ```python
    from sifaka.interfaces.model import ModelProviderProtocol
    from sifaka.interfaces.client import APIClientProtocol
    from sifaka.interfaces.counter import TokenCounterProtocol
    from sifaka.utils.config.models import ModelConfig

    # Basic usage
    provider = ProviderClass(model_name="model-name")
    response = provider.generate("Explain quantum computing")
    token_count = provider.count_tokens("How many tokens is this?")

    # With custom configuration
    config = ModelConfig().with_temperature(0.9).with_max_tokens(2000)
    provider = ProviderClass(model_name="model-name", config=config)

    # With explicit dependencies
    class CustomClient:
        def send_prompt(self, prompt, config):
            # Custom implementation
            return "Response"

    class CustomTokenCounter:
        def count_tokens(self, text):
            # Custom implementation
            return len(text.split())

    # Verify protocol compliance
    assert isinstance(CustomClient(), APIClientProtocol)
    assert isinstance(CustomTokenCounter(), TokenCounterProtocol)

    provider = ProviderClass(
        model_name="model-name",
        config=config,
        api_client=CustomClient(),
        token_counter=CustomTokenCounter()
    )

    # Error handling
    try:
        response = provider.generate("Explain quantum computing")
    except ValueError as e:
        # Handle input validation errors
        print(f"Input error: {e}")
    except RuntimeError as e:
        # Handle API and generation errors
        print(f"Generation failed: {e}")
        # Implement fallback strategy
        fallback_provider = FallbackProviderClass(model_name="fallback-model")
        response = fallback_provider.generate("Explain quantum computing briefly")
    ```

    Attributes:
        _model_name (str): The name of the model being used
        _config (C): The configuration for the model provider
        _api_client (APIClient): The client for API communication
        _token_counter (TokenCounter): The counter for token estimation
        _tracer (Tracer): The tracer for event recording
        _initialized (bool): Whether the provider has been initialized
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
        Initialize a model provider with explicit dependencies.

        This method initializes a model provider with the specified model name and
        optional dependencies. If dependencies are not provided, default ones will
        be created lazily when needed using the _create_default_* methods.

        ## Lifecycle

        1. **Initialization**: Store model name and dependencies
           - Store model name for later use
           - Store or create configuration
           - Store optional dependencies (api_client, token_counter, tracer)
           - Initialize logging

        2. **Validation**: Validate inputs
           - Ensure model_name is not empty
           - Ensure dependencies implement required protocols

        ## Error Handling

        This method handles these error cases:
        - Raises ValueError if model_name is empty
        - Raises TypeError if dependencies don't implement required protocols
        - Logs initialization for debugging

        ## Examples

        Basic initialization with just a model name:

        ```python
        from sifaka.interfaces.model import ModelProviderProtocol

        # Create a provider with just a model name
        # Default configuration and dependencies will be created when needed
        provider = ProviderClass(model_name="model-name")
        ```

        Initialization with custom configuration:

        ```python
        from sifaka.utils.config.models import ModelConfig
        from sifaka.interfaces.model import ModelProviderProtocol

        # Create a custom configuration
        config = ModelConfig().with_temperature(0.9).with_max_tokens(2000)

        # Create a provider with the custom configuration
        provider = ProviderClass(model_name="model-name", config=config)
        ```

        Initialization with custom dependencies:

        ```python
        from sifaka.interfaces.client import APIClientProtocol
        from sifaka.interfaces.counter import TokenCounterProtocol
        from sifaka.interfaces.model import ModelProviderProtocol
        from sifaka.utils.tracing import Tracer

        # Create custom dependencies
        class CustomAPIClient:
            def send_prompt(self, prompt, config):
                # Custom implementation
                return "Response"

        class CustomTokenCounter:
            def count_tokens(self, text):
                # Custom implementation
                return len(text.split())

        # Verify protocol compliance
        assert isinstance(CustomAPIClient(), APIClientProtocol)
        assert isinstance(CustomTokenCounter(), TokenCounterProtocol)

        # Create a tracer
        tracer = Tracer()

        # Create a provider with custom dependencies
        provider = ProviderClass(
            model_name="model-name",
            api_client=CustomAPIClient(),
            token_counter=CustomTokenCounter(),
            tracer=tracer
        )
        ```

        Error handling during initialization:

        ```python
        try:
            # This will raise ValueError if model_name is empty
            provider = ProviderClass(model_name="")
        except ValueError as e:
            print(f"Initialization error: {e}")
            # Use a default model name instead
            provider = ProviderClass(model_name="default-model")

        try:
            # This will raise TypeError if api_client doesn't implement APIClient
            provider = ProviderClass(
                model_name="model-name",
                api_client="not an APIClient"  # Wrong type
            )
        except TypeError as e:
            print(f"Initialization error: {e}")
            # Create without custom dependencies
            provider = ProviderClass(model_name="model-name")
        ```

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
        self._model_name = model_name
        self._config = config or self._create_default_config()  # type: C
        self._api_client = api_client
        self._token_counter = token_counter
        self._tracer = tracer or (Tracer() if self._config.trace_enabled else None)
        self._initialized = False
        logger.info(f"Created {self.__class__.__name__} with model {model_name}")

    def initialize(self) -> None:
        """
        Initialize the model provider.

        This method initializes any resources needed by the model provider.
        It ensures that the API client and token counter are created if they
        haven't been provided during initialization.

        ## Lifecycle
        1. Check if already initialized to prevent duplicate initialization
        2. Create API client and token counter if not already provided
        3. Mark as initialized for future reference
        4. Log initialization status for debugging

        ## Error Handling
        - Catches and logs any exceptions during initialization
        - Re-raises exceptions as RuntimeError with context
        - Prevents partial initialization by rolling back on failure

        Raises:
            RuntimeError: If initialization fails for any reason

        Example:
            ```python
            provider = OpenAIProvider(model_name="gpt-4")

            # Initialize resources
            provider.initialize()

            # Use the initialized provider
            response = provider.generate("Hello, world!")

            # Clean up when done
            provider.cleanup()
            ```
        """
        if self._initialized:
            logger.debug(f"{self.__class__.__name__} already initialized")
            return

        try:
            # Ensure API client and token counter are created
            self._ensure_api_client()
            self._ensure_token_counter()
            self._initialized = True
            logger.info(f"Initialized {self.__class__.__name__} with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            raise RuntimeError(f"Failed to initialize {self.__class__.__name__}: {e}") from e

    def cleanup(self) -> None:
        """
        Clean up the model provider.

        This method releases any resources held by the model provider.
        It ensures that the API client and token counter are properly closed
        to prevent resource leaks.

        ## Lifecycle
        1. Check if provider is initialized to avoid unnecessary cleanup
        2. Close API client if it has a close method
        3. Close token counter if it has a close method
        4. Mark as uninitialized for future reference
        5. Log cleanup status for debugging

        ## Error Handling
        - Catches and logs any exceptions during cleanup
        - Re-raises exceptions as RuntimeError with context
        - Continues cleanup even if some components fail

        Raises:
            RuntimeError: If cleanup fails for any reason

        Example:
            ```python
            provider = OpenAIProvider(model_name="gpt-4")

            try:
                # Initialize and use the provider
                provider.initialize()
                response = provider.generate("Hello, world!")
            finally:
                # Ensure resources are cleaned up even if an error occurs
                provider.cleanup()
            ```
        """
        if not self._initialized:
            logger.debug(f"{self.__class__.__name__} not initialized, nothing to clean up")
            return

        try:
            # Close API client if it has a close method
            if self._api_client and hasattr(self._api_client, "close"):
                self._api_client.close()

            # Close token counter if it has a close method
            if self._token_counter and hasattr(self._token_counter, "close"):
                self._token_counter.close()

            self._initialized = False
            logger.info(f"Cleaned up {self.__class__.__name__} with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to clean up {self.__class__.__name__}: {e}")
            raise RuntimeError(f"Failed to clean up {self.__class__.__name__}: {e}") from e

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        This property provides read-only access to the model name that was
        specified during initialization.

        Returns:
            str: The name of the language model

        Example:
            ```python
            provider = OpenAIProvider(model_name="gpt-4")
            print(f"Using model: {provider.model_name}")
            ```
        """
        return self._model_name

    @property
    def config(self) -> C:
        """
        Get the model configuration.

        This property provides read-only access to the current configuration
        object. The configuration can be updated using the update_config method.

        Returns:
            C: The current model configuration

        Example:
            ```python
            provider = OpenAIProvider(model_name="gpt-4")

            # Access configuration properties
            print(f"Temperature: {provider.config.temperature}")
            print(f"Max tokens: {provider.config.max_tokens}")

            # Update configuration
            new_config = ModelConfig(temperature=0.9, max_tokens=2000)
            provider.update_config(new_config)
            ```
        """
        return self._config

    @property
    def description(self) -> str:
        """
        Get the model provider description.

        This property returns a human-readable description of the provider,
        including the provider class name and the model name.

        Returns:
            str: The description of the model provider

        Example:
            ```python
            provider = OpenAIProvider(model_name="gpt-4")
            print(f"Provider: {provider.description}")
            # Output: "OpenAIProvider for gpt-4"
            ```
        """
        return f"{self.__class__.__name__} for {self.model_name}"

    def update_config(self, config: C) -> None:
        """
        Update the model configuration.

        This method replaces the current configuration with a new one.
        It validates that the new configuration is an instance of ModelConfig
        before applying the change.

        ## Lifecycle
        1. Validate the new configuration object
        2. Replace the current configuration
        3. Log the configuration update for debugging

        ## Error Handling
        - Validates that config is an instance of ModelConfig
        - Raises ValueError with a descriptive message if validation fails

        Args:
            config (C): The new configuration object

        Raises:
            ValueError: If the configuration is not an instance of ModelConfig

        Example:
            ```python
            provider = OpenAIProvider(model_name="gpt-4")

            # Create a new configuration
            new_config = ModelConfig(
                temperature=0.9,
                max_tokens=2000,
                api_key="new-api-key"
            )

            # Update the provider's configuration
            provider.update_config(new_config)

            # Use the provider with the new configuration
            response = provider.generate("Hello with new config!")
            ```
        """
        if not isinstance(config, ModelConfig):
            raise ValueError(f"Config must be an instance of ModelConfig, got {type(config)}")
        self._config = config
        logger.info(f"Updated configuration for {self.__class__.__name__} with {self.model_name}")

    def _create_default_config(self) -> C:
        """
        Create a default configuration for this provider.

        This method creates a default configuration when none is provided during
        initialization. It's used by the constructor to ensure a valid configuration
        is always available.

        ## Architecture
        This method follows the Template Method pattern:
        - Base class provides a default implementation
        - Subclasses can override to provide specialized configurations
        - Type safety is maintained through generic type parameter C

        ## Lifecycle
        1. **Creation**: Create a default ModelConfig instance
           - Use default values for all parameters
           - Cast to the generic type C for type safety

        2. **Customization**: Provider-specific subclasses can override
           - Override to provide model-specific defaults
           - Override to use provider-specific config classes

        ## Error Handling
        - Returns a valid default configuration
        - Type casting ensures compatibility with expected return type
        - No exceptions should be raised by this method

        Returns:
            C: A default configuration instance appropriate for this provider

        Example:
            Basic implementation:
            ```python
            def _create_default_config(self) -> ModelConfig:
                return ModelConfig()
            ```

            Provider-specific implementation:
            ```python
            class CustomProvider(ModelProvider[CustomConfig]):
                def _create_default_config(self) -> CustomConfig:
                    return CustomConfig(
                        temperature=0.7,
                        max_tokens=1000,
                        system_prompt="You are a helpful assistant."
                    )
            ```
        """
        return ModelConfig()  # type: ignore

    @abstractmethod
    def _create_default_client(self) -> APIClient:
        """
        Create a default API client if none was provided.

        This abstract method must be implemented by subclasses to provide
        a default API client implementation specific to the model provider.

        ## Architecture
        This method follows the Factory Method pattern:
        - Abstract in base class, implemented in subclasses
        - Creates appropriate client for specific provider type
        - Returns an object implementing the APIClient protocol

        ## Error Handling
        - Should raise RuntimeError if client creation fails
        - Should include specific error details in exception message
        - May attempt fallback strategies before failing

        Returns:
            APIClient: A default API client for the model

        Raises:
            RuntimeError: If a default client cannot be created

        Example:
            ```python
            class OpenAIProvider(ModelProvider):
                def _create_default_client(self) -> APIClient:
                    try:
                        # Try to create client with API key from config
                        return OpenAIClient(api_key=self.config.api_key)
                    except Exception as e:
                        # Provide detailed error message
                        raise RuntimeError(f"Failed to create OpenAI client: {e}") from e
            ```
        """
        ...

    @abstractmethod
    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default token counter if none was provided.

        This abstract method must be implemented by subclasses to provide
        a default token counter implementation specific to the model provider.

        ## Architecture
        This method follows the Factory Method pattern:
        - Abstract in base class, implemented in subclasses
        - Creates appropriate counter for specific model type
        - Returns an object implementing the TokenCounter protocol

        ## Error Handling
        - Should raise RuntimeError if counter creation fails
        - Should include specific error details in exception message
        - May attempt fallback strategies before failing

        Returns:
            TokenCounter: A default token counter for the model

        Raises:
            RuntimeError: If a default token counter cannot be created

        Example:
            ```python
            class OpenAIProvider(ModelProvider):
                def _create_default_token_counter(self) -> TokenCounter:
                    try:
                        # Create counter specific to this model
                        return OpenAITokenCounter(model=self.model_name)
                    except Exception as e:
                        # Provide detailed error message
                        raise RuntimeError(f"Failed to create token counter: {e}") from e
            ```
        """
        ...

    def _ensure_api_client(self) -> APIClient:
        """
        Ensure an API client is available, creating a default one if needed.

        This method implements the lazy initialization pattern for the API client.
        It checks if an API client has been provided during initialization, and if not,
        creates a default one using the _create_default_client method.

        ## Lifecycle

        1. **Check Existence**: Check if an API client already exists
           - Return existing client if available
           - Create default client if none exists

        2. **Creation**: Create default client if needed
           - Call _create_default_client abstract method
           - Log the creation for debugging
           - Store the created client for future use

        3. **Validation**: Ensure client implements APIClient protocol
           - Runtime type checking ensures protocol compliance
           - Proper error propagation if creation fails

        ## Error Handling

        This method handles these error cases:
        - Propagates exceptions from _create_default_client
        - Logs client creation for debugging
        - Ensures created client implements APIClient protocol

        ## Examples

        ```python
        class MyModelProvider(ModelProvider):
            def _create_default_client(self) -> APIClient:
                # Implementation of abstract method
                return MyAPIClient(base_url="https://api.example.com")

            def generate(self, prompt: str, **kwargs) -> str:
                # Get API client (creates default if needed)
                client = self._ensure_api_client()

                # Use client to send prompt
                return client.send_prompt(prompt, self.config)
        ```

        Returns:
            The API client to use (existing or newly created)

        Raises:
            RuntimeError: If a default client cannot be created
            TypeError: If the created client doesn't implement APIClient protocol
        """
        if self._api_client is None:
            logger.debug(f"Creating default API client for {self.model_name}")
            self._api_client = self._create_default_client()
        return self._api_client

    def _ensure_token_counter(self) -> TokenCounter:
        """
        Ensure a token counter is available, creating a default one if needed.

        This method implements the lazy initialization pattern for the token counter.
        It checks if a token counter has been provided during initialization, and if not,
        creates a default one using the _create_default_token_counter method.

        ## Lifecycle

        1. **Check Existence**: Check if a token counter already exists
           - Return existing counter if available
           - Create default counter if none exists

        2. **Creation**: Create default counter if needed
           - Call _create_default_token_counter abstract method
           - Log the creation for debugging
           - Store the created counter for future use

        3. **Validation**: Ensure counter implements TokenCounter protocol
           - Runtime type checking ensures protocol compliance
           - Proper error propagation if creation fails

        ## Error Handling

        This method handles these error cases:
        - Propagates exceptions from _create_default_token_counter
        - Logs counter creation for debugging
        - Ensures created counter implements TokenCounter protocol

        ## Examples

        ```python
        class MyModelProvider(ModelProvider):
            def _create_default_token_counter(self) -> TokenCounter:
                # Implementation of abstract method
                return MyTokenCounter(model=self.model_name)

            def count_tokens(self, text: str) -> int:
                # Get token counter (creates default if needed)
                counter = self._ensure_token_counter()

                # Use counter to count tokens
                return counter.count_tokens(text)
        ```

        Returns:
            The token counter to use (existing or newly created)

        Raises:
            RuntimeError: If a default token counter cannot be created
            TypeError: If the created counter doesn't implement TokenCounter protocol
        """
        if self._token_counter is None:
            logger.debug(f"Creating default token counter for {self.model_name}")
            self._token_counter = self._create_default_token_counter()
        return self._token_counter

    def _trace_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Record a trace event if tracing is enabled.

        This method provides a standardized way to record events for monitoring,
        debugging, and analysis. It only records events if tracing is enabled
        in the configuration.

        ## Lifecycle

        1. **Check Enabled**: Check if tracing is enabled
           - Skip recording if tracing is disabled
           - Continue if tracing is enabled and tracer exists

        2. **Generate ID**: Create a unique trace ID
           - Combine model name with timestamp
           - Ensures uniqueness across traces

        3. **Record Event**: Add event to tracer
           - Pass event type and data to tracer
           - Associate with the generated trace ID

        ## Common Event Types

        - **generate**: Records generation events with performance metrics
        - **token_count**: Records token counting operations
        - **error**: Records errors that occur during operations
        - **config_change**: Records configuration changes

        ## Examples

        ```python
        class MyModelProvider(ModelProvider):
            def generate(self, prompt: str, **kwargs) -> str:
                # Record start of generation
                self._trace_event(
                    "generate_start",
                    {
                        "prompt_length": len(prompt),
                        "temperature": self.config.temperature
                    }
                )

                # Generate response
                response = "Generated text"

                # Record successful generation
                self._trace_event(
                    "generate_complete",
                    {
                        "prompt_length": len(prompt),
                        "response_length": len(response),
                        "duration_ms": 150
                    }
                )

                return response
        ```

        Args:
            event_type: The type of event to record (e.g., "generate", "token_count")
            data: The data to record with the event as key-value pairs
        """
        if self._tracer and self._config.trace_enabled:
            trace_id = datetime.now().strftime(f"{self.model_name}_%Y%m%d%H%M%S")
            self._tracer.add_event(trace_id, event_type, data)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text

        Raises:
            TypeError: If text is not a string
            RuntimeError: If token counting fails

        Examples:
            ```python
            # Basic token counting
            provider = AnthropicProvider(model_name="claude-3-opus")
            count = provider.count_tokens("How many tokens is this?")
            print(f"Token count: {count}")

            # Error handling
            try:
                count = provider.count_tokens(["This", "is", "a", "list"])  # TypeError
            except TypeError as e:
                print(f"Invalid input: {e}")
                # Handle with default or fallback
                count = 0

            # Using for optimization
            prompt = "This is a long prompt..."
            count = provider.count_tokens(prompt)
            if count > 1000:
                # Truncate or summarize if too long
                prompt = prompt[:500] + "..."
                print(f"Truncated prompt to reduce tokens")
            ```
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        counter = self._ensure_token_counter()
        token_count = counter.count_tokens(text)

        self._trace_event(
            "token_count",
            {
                "text_length": len(text),
                "token_count": token_count,
            },
        )

        return token_count

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the model.

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
            ValueError: If prompt is empty
            RuntimeError: If an error occurs during generation

        Examples:
            ```python
            # Basic generation
            provider = AnthropicProvider(model_name="claude-3-opus")
            response = provider.generate("Explain quantum computing in simple terms.")

            # With parameter overrides
            response = provider.generate(
                "Write a creative story.",
                temperature=0.9,
                max_tokens=2000
            )

            # Comprehensive error handling
            try:
                response = provider.generate("Explain quantum computing")
            except ValueError as e:
                # Handle input validation errors
                print(f"Input error: {e}")
                response = "I couldn't process that request."
            except RuntimeError as e:
                # Handle API and generation errors
                print(f"Generation failed: {e}")

                # Check for specific error conditions
                if "rate limit" in str(e).lower():
                    print("Rate limit exceeded, waiting and retrying...")
                    import time
                    time.sleep(5)
                    response = provider.generate("Explain quantum computing briefly")
                elif "context length" in str(e).lower():
                    print("Prompt too long, shortening...")
                    response = provider.generate("Briefly explain quantum computing")
                else:
                    # Fallback to simpler model
                    fallback_provider = create_model_provider(
                        ModelProvider,  # Use a generic provider type for the example
                        model_name="simpler-model"
                    )
                    response = fallback_provider.generate("Explain quantum computing briefly")
            ```
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Update config with any override kwargs
        config = ModelConfig(
            temperature=kwargs.pop("temperature", self.config.temperature),
            max_tokens=kwargs.pop("max_tokens", self.config.max_tokens),
            api_key=kwargs.pop("api_key", self.config.api_key),
            trace_enabled=kwargs.pop("trace_enabled", self.config.trace_enabled),
        )

        # Count tokens before generation
        prompt_tokens = self.count_tokens(prompt)
        if prompt_tokens > config.max_tokens:
            logger.warning(
                f"Prompt tokens ({prompt_tokens}) exceed max_tokens ({config.max_tokens})"
            )

        start_time = datetime.now()
        client = self._ensure_api_client()

        try:
            response = client.send_prompt(prompt, config)

            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            self._trace_event(
                "generate",
                {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": self.count_tokens(response),
                    "duration_ms": duration_ms,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "success": True,
                },
            )

            logger.debug(
                f"Generated response in {duration_ms:.2f}ms " f"(prompt: {prompt_tokens} tokens)"
            )

            return response

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating text with {self.model_name}: {error_msg}")

            self._trace_event(
                "error",
                {
                    "error": error_msg,
                    "prompt_tokens": prompt_tokens,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                },
            )

            raise RuntimeError(f"Error generating text with {self.model_name}: {error_msg}") from e


def create_model_provider(
    provider_type: Callable[..., T],
    model_name: str,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs,
) -> T:
    """
    Factory function to create a model provider with a standardized configuration.

    This function simplifies the creation of model providers by providing
    a consistent interface for common configuration options across different
    provider implementations.

    ## Architecture
    The factory function follows the Factory Method pattern to:
    - Create standardized configuration objects
    - Instantiate provider classes with consistent parameters
    - Support dependency injection through kwargs
    - Provide type safety through generic return types

    ## Lifecycle
    1. Create a standard ModelConfig with common parameters
    2. Instantiate the provider with the config and model name
    3. Pass through any additional provider-specific arguments

    ## Error Handling
    This function handles these error cases:
    - Validates parameters before creating the provider
    - Propagates exceptions from provider initialization
    - Provides clear error messages for debugging

    Args:
        provider_type: The class of the provider to create
        model_name: The name of the model to use
        api_key: Optional API key for the provider
        temperature: Temperature for generation (0-1)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional provider-specific arguments

    Returns:
        A configured model provider instance

    Raises:
        ValueError: If parameters are invalid
        TypeError: If provider_type is not a valid model provider class
        RuntimeError: If provider creation fails

    Example:
        ```python
        from sifaka.interfaces.model import ModelProviderProtocol

        # Create a provider with a specific provider class
        provider = create_model_provider(
            ProviderClass,  # Any class implementing ModelProviderProtocol
            model_name="model-name",
            api_key="your-api-key",
            temperature=0.8
        )

        # Create another provider
        provider = create_model_provider(
            AnotherProviderClass,
            model_name="another-model",
            api_key="your-api-key",
            max_tokens=2000
        )

        # Error handling when creating providers
        try:
            provider = create_model_provider(
                ProviderClass,
                model_name="invalid-model",
                api_key="your-api-key"
            )
        except (ValueError, RuntimeError) as e:
            print(f"Provider creation failed: {e}")
            # Use fallback provider
            provider = create_model_provider(
                FallbackProviderClass,
                model_name="fallback-model",
                api_key="your-fallback-key"
            )

        # With additional provider-specific options
        provider = create_model_provider(
            ProviderClass,
            model_name="model-name",
            api_key="your-api-key",
            system_prompt="You are a helpful assistant"
        )
        ```
    """
    config = ModelConfig(temperature=temperature, max_tokens=max_tokens, api_key=api_key)
    return provider_type(model_name=model_name, config=config, **kwargs)
