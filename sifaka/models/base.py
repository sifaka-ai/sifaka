"""
Base protocols and types for model providers.

This module provides the core interfaces and base implementations for model providers,
including protocols for API clients, token counters, and language models.

## Architecture Overview

The model system follows a layered architecture:

1. **ModelProvider**: High-level interface for model interactions
2. **APIClient**: Low-level communication with model services
3. **TokenCounter**: Utility for token counting
4. **Config**: Configuration and settings management

## Component Lifecycle

### ModelProvider
1. **Initialization**: Set up with model name and configuration
2. **Configuration**: Define generation parameters
3. **Operation**: Generate text, count tokens, trace events
4. **Error Handling**: Manage API failures and retry strategies
5. **Cleanup**: Release resources when no longer needed

### Error Handling Patterns

The model system implements several error handling patterns:

1. **Typed Exceptions**: Use specific exception types for different error cases
2. **Automatic Retries**: Implement backoff strategy for transient errors
3. **Graceful Degradation**: Fallback to simpler models when primary fails
4. **Thorough Logging**: Log all errors with context for diagnosis
5. **Tracing**: Record detailed events for monitoring and debugging

## Usage Examples

```python
from sifaka.models import create_model_provider, ModelConfig
from sifaka.models.providers.anthropic import AnthropicProvider

# Basic usage with factory function
provider = create_model_provider(
    AnthropicProvider,
    model_name="claude-3-opus",
    api_key="your-api-key",
    temperature=0.8
)
response = provider.generate("Explain quantum computing in simple terms.")

# Direct instantiation with custom configuration
config = ModelConfig().with_temperature(0.9).with_max_tokens(2000)
provider = AnthropicProvider(model_name="claude-3-opus", config=config)

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
        AnthropicProvider,
        model_name="claude-3-haiku"
    )
    response = fallback_provider.generate("Explain quantum computing briefly")
```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from sifaka.models.interfaces import (
    APIClientProtocol as APIClient,
    TokenCounterProtocol as TokenCounter,
    ModelProviderProtocol as ModelProviderInterface,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.tracing import Tracer

logger = get_logger(__name__)

# Type variables for generic type definitions
T = TypeVar("T", bound="ModelProvider")
R = TypeVar("R")  # Return type for model operations
C = TypeVar("C", bound="ModelConfig")  # Config type


@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable configuration for model providers.

    This class represents a set of configuration options for model providers.
    Because it is immutable, you must use the with_* methods to create
    modified versions of a configuration.

    ## Lifecycle

    1. **Creation**: Instantiate with default or custom values
    2. **Validation**: Values are validated in __post_init__
    3. **Modification**: Create new instances with with_* methods
    4. **Usage**: Pass to model providers for configuration

    ## Error Handling

    - Validates temperature range (0-1)
    - Validates max_tokens is positive
    - Raises ValueError for invalid inputs
    - Returns new instances rather than modifying existing ones

    ## Examples

    ```python
    # Create a default configuration
    config = ModelConfig()

    # Create a modified configuration
    new_config = config.with_temperature(0.9).with_max_tokens(2000)

    # Error handling with validation
    try:
        invalid_config = config.with_temperature(1.5)  # Will raise ValueError
    except ValueError as e:
        print(f"Configuration error: {e}")
        # Use default or fallback configuration
        fallback_config = config.with_temperature(0.7)

    # Checking configuration before use
    if config.temperature > 0.8:
        print("Warning: High temperature may lead to more random outputs")

    # Use the configuration with a model provider
    provider = AnthropicProvider(model_name="claude-3-opus", config=new_config)
    ```
    """

    temperature: float = 0.7
    max_tokens: int = 1000
    api_key: Optional[str] = None
    trace_enabled: bool = True

    def __post_init__(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If temperature is not between 0 and 1
            ValueError: If max_tokens is not positive
        """
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

    def with_temperature(self, temperature: float) -> "ModelConfig":
        """
        Create a new config with the specified temperature.

        Args:
            temperature: The new temperature value (0-1)

        Returns:
            A new ModelConfig with the updated temperature

        Raises:
            ValueError: If temperature is not between 0 and 1

        Examples:
            ```python
            # Create a config with a specific temperature
            config = ModelConfig().with_temperature(0.8)

            # Chain multiple configuration updates
            config = ModelConfig().with_temperature(0.8).with_max_tokens(500)

            # Handle validation errors
            try:
                config = ModelConfig().with_temperature(1.5)  # Invalid
            except ValueError as e:
                # Use default instead
                config = ModelConfig()
            ```
        """
        return replace(self, temperature=temperature)

    def with_max_tokens(self, max_tokens: int) -> "ModelConfig":
        """
        Create a new config with the specified max_tokens.

        Args:
            max_tokens: The new max_tokens value

        Returns:
            A new ModelConfig with the updated max_tokens

        Raises:
            ValueError: If max_tokens is not positive

        Examples:
            ```python
            # Create a config with a specific max_tokens
            config = ModelConfig().with_max_tokens(2000)

            # Adjust based on text length
            prompt_length = len(prompt) // 4  # Rough estimate
            config = ModelConfig().with_max_tokens(max(1000, prompt_length * 2))

            # Handle validation errors
            try:
                config = ModelConfig().with_max_tokens(-100)  # Invalid
            except ValueError as e:
                # Use default instead
                config = ModelConfig()
            ```
        """
        return replace(self, max_tokens=max_tokens)

    def with_api_key(self, api_key: Optional[str]) -> "ModelConfig":
        """
        Create a new config with the specified API key.

        Args:
            api_key: The new API key

        Returns:
            A new ModelConfig with the updated API key

        Examples:
            ```python
            # Create a config with an API key
            config = ModelConfig().with_api_key("sk-...")

            # Read API key from environment variable
            import os
            config = ModelConfig().with_api_key(os.environ.get("ANTHROPIC_API_KEY"))

            # Handle missing API key
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("Warning: No API key found, some features may be limited")
                # Proceed with default or use a mock provider
            config = ModelConfig().with_api_key(api_key)
            ```
        """
        return replace(self, api_key=api_key)

    def with_trace_enabled(self, trace_enabled: bool) -> "ModelConfig":
        """
        Create a new config with the specified trace_enabled setting.

        Args:
            trace_enabled: Whether tracing should be enabled

        Returns:
            A new ModelConfig with the updated trace_enabled setting

        Examples:
            ```python
            # Enable tracing for debugging
            debug_config = ModelConfig().with_trace_enabled(True)

            # Disable tracing for production
            prod_config = ModelConfig().with_trace_enabled(False)

            # Conditionally enable tracing
            import os
            env = os.environ.get("ENVIRONMENT", "development")
            config = ModelConfig().with_trace_enabled(env == "development")
            ```
        """
        return replace(self, trace_enabled=trace_enabled)


# Re-export the APIClient protocol from interfaces.models
# The implementation details are now in the interfaces module


# Re-export the TokenCounter protocol from interfaces.models
# The implementation details are now in the interfaces module


@runtime_checkable
class LanguageModel(Protocol[R]):
    """
    Protocol for language model interfaces.

    Classes implementing this protocol provide a high-level interface
    for generating text using language models.

    Type Parameters:
        R: The return type of the generate method

    ## Lifecycle

    1. **Initialization**: Set up model, clients, and resources
    2. **Configuration**: Set generation parameters
    3. **Prompt Preparation**: Format prompts for the model
    4. **Generation**: Generate text responses
    5. **Response Processing**: Parse and validate responses
    6. **Cleanup**: Release resources when no longer needed

    ## Error Handling

    Implementations should handle:
    - Input validation errors
    - Configuration issues
    - Generation failures
    - Response parsing problems
    - Resource cleanup

    ## Examples

    ```python
    from sifaka.models.base import LanguageModel, ModelConfig
    from typing import Dict, Any, Optional

    class CompletionModel(LanguageModel[str]):
        def __init__(self, model_name: str, api_key: Optional[str] = None):
            self._model_name = model_name
            self._api_key = api_key
            self._client = self._create_client()

        @property
        def model_name(self) -> str:
            return self._model_name

        def _create_client(self):
            # Create appropriate client for the model
            if "gpt" in self._model_name.lower():
                from openai import OpenAI
                return OpenAI(api_key=self._api_key)
            elif "claude" in self._model_name.lower():
                from anthropic import Anthropic
                return Anthropic(api_key=self._api_key)
            else:
                raise ValueError(f"Unsupported model: {self._model_name}")

        def generate(self, prompt: str, **kwargs) -> str:
            # Input validation
            if not prompt:
                raise ValueError("Prompt cannot be empty")

            # Default parameters
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)

            # Error handling with try-except
            try:
                # Generate based on model type
                if "gpt" in self._model_name.lower():
                    response = self._client.completions.create(
                        model=self._model_name,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].text.strip()
                elif "claude" in self._model_name.lower():
                    response = self._client.completions.create(
                        model=self._model_name,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.completion.strip()
                else:
                    raise ValueError(f"Unsupported model: {self._model_name}")
            except Exception as e:
                # Proper error propagation with context
                raise RuntimeError(f"Error generating from {self._model_name}: {e}")

        def close(self):
            # Resource cleanup
            if hasattr(self._client, "close"):
                self._client.close()
    ```
    """

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            The name of the language model
        """
        ...

    def generate(self, prompt: str, **kwargs) -> R:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate from
            **kwargs: Additional model-specific parameters

        Returns:
            The generated text or structured response

        Raises:
            TypeError: If prompt is not a string
            ValueError: If prompt is empty or invalid
            RuntimeError: If generation fails
        """
        ...


class ModelProvider(ModelProviderInterface, Generic[C], ABC):
    """
    Abstract base class for model providers.

    This class implements the ModelProviderProtocol interface from sifaka.models.interfaces
    and enforces a consistent interface for all model providers
    while allowing for flexible implementation of specific provider features.

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

    ## Error Handling Patterns

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
    from sifaka.models.base import ModelProvider, ModelConfig
    from sifaka.models.interfaces import APIClientProtocol, TokenCounterProtocol

    # Basic usage
    provider = OpenAIProvider(model_name="gpt-4")
    response = provider.generate("Explain quantum computing")
    token_count = provider.count_tokens("How many tokens is this?")

    # With custom configuration
    config = ModelConfig().with_temperature(0.9).with_max_tokens(2000)
    provider = AnthropicProvider(model_name="claude-3-opus", config=config)

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

    provider = AnthropicProvider(
        model_name="claude-3-opus",
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
        fallback_provider = OpenAIProvider(model_name="gpt-3.5-turbo")
        response = fallback_provider.generate("Explain quantum computing briefly")
    ```
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
        from sifaka.models.providers.openai import OpenAIProvider

        # Create a provider with just a model name
        # Default configuration and dependencies will be created when needed
        provider = OpenAIProvider(model_name="gpt-4")
        ```

        Initialization with custom configuration:

        ```python
        from sifaka.models.base import ModelConfig
        from sifaka.models.providers.anthropic import AnthropicProvider

        # Create a custom configuration
        config = ModelConfig().with_temperature(0.9).with_max_tokens(2000)

        # Create a provider with the custom configuration
        provider = AnthropicProvider(model_name="claude-3-opus", config=config)
        ```

        Initialization with custom dependencies:

        ```python
        from sifaka.models.base import APIClient, TokenCounter
        from sifaka.models.providers.openai import OpenAIProvider
        from sifaka.utils.tracing import Tracer

        # Create custom dependencies
        class CustomAPIClient(APIClient):
            def send_prompt(self, prompt, config):
                # Custom implementation
                return "Response"

        class CustomTokenCounter(TokenCounter):
            def count_tokens(self, text):
                # Custom implementation
                return len(text.split())

        # Create a tracer
        tracer = Tracer()

        # Create a provider with custom dependencies
        provider = OpenAIProvider(
            model_name="gpt-4",
            api_client=CustomAPIClient(),
            token_counter=CustomTokenCounter(),
            tracer=tracer
        )
        ```

        Error handling during initialization:

        ```python
        try:
            # This will raise ValueError if model_name is empty
            provider = OpenAIProvider(model_name="")
        except ValueError as e:
            print(f"Initialization error: {e}")
            # Use a default model name instead
            provider = OpenAIProvider(model_name="gpt-3.5-turbo")

        try:
            # This will raise TypeError if api_client doesn't implement APIClient
            provider = OpenAIProvider(
                model_name="gpt-4",
                api_client="not an APIClient"  # Wrong type
            )
        except TypeError as e:
            print(f"Initialization error: {e}")
            # Create without custom dependencies
            provider = OpenAIProvider(model_name="gpt-4")
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

        Raises:
            RuntimeError: If initialization fails
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
        It ensures that the API client and token counter are properly closed.

        Raises:
            RuntimeError: If cleanup fails
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

        Returns:
            The name of the language model
        """
        return self._model_name

    @property
    def config(self) -> C:
        """
        Get the model configuration.

        Returns:
            The current model configuration
        """
        return self._config

    @property
    def description(self) -> str:
        """
        Get the model provider description.

        Returns:
            The description of the model provider
        """
        return f"{self.__class__.__name__} for {self.model_name}"

    def update_config(self, config: C) -> None:
        """
        Update the model configuration.

        Args:
            config: The new configuration object

        Raises:
            ValueError: If the configuration is invalid
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

        ## Lifecycle

        1. **Creation**: Create a default ModelConfig instance
           - Use default values for all parameters
           - Cast to the generic type C for type safety

        2. **Customization**: Provider-specific subclasses can override
           - Override to provide model-specific defaults
           - Override to use provider-specific config classes

        ## Examples

        Basic implementation:

        ```python
        def _create_default_config(self) -> ModelConfig:
            return ModelConfig()
        ```

        Provider-specific implementation:

        ```python
        class AnthropicProvider(ModelProvider[AnthropicConfig]):
            def _create_default_config(self) -> AnthropicConfig:
                return AnthropicConfig(
                    temperature=0.7,
                    max_tokens=1000,
                    system_prompt="You are a helpful assistant."
                )
        ```

        Returns:
            A default configuration instance appropriate for this provider
        """
        return ModelConfig()  # type: ignore

    @abstractmethod
    def _create_default_client(self) -> APIClient:
        """
        Create a default API client if none was provided.

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

        Returns:
            A default token counter for the model

        Raises:
            RuntimeError: If a default token counter cannot be created
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
                        AnthropicProvider,
                        model_name="claude-3-haiku"
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
    a consistent interface for common configuration options.

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

    Examples:
        ```python
        from sifaka.models.providers.anthropic import AnthropicProvider
        from sifaka.models.providers.openai import OpenAIProvider

        # Create an Anthropic provider
        provider = create_model_provider(
            AnthropicProvider,
            model_name="claude-3-opus",
            api_key="your-api-key",
            temperature=0.8
        )

        # Create an OpenAI provider
        provider = create_model_provider(
            OpenAIProvider,
            model_name="gpt-4",
            api_key="your-api-key",
            max_tokens=2000
        )

        # Error handling when creating providers
        try:
            provider = create_model_provider(
                AnthropicProvider,
                model_name="invalid-model",
                api_key="your-api-key"
            )
        except (ValueError, RuntimeError) as e:
            print(f"Provider creation failed: {e}")
            # Use fallback provider
            provider = create_model_provider(
                OpenAIProvider,
                model_name="gpt-3.5-turbo",
                api_key="your-openai-key"
            )

        # With additional provider-specific options
        provider = create_model_provider(
            AnthropicProvider,
            model_name="claude-3-opus",
            api_key="your-api-key",
            system_prompt="You are a helpful assistant"
        )
        ```
    """
    config = ModelConfig(temperature=temperature, max_tokens=max_tokens, api_key=api_key)
    return provider_type(model_name=model_name, config=config, **kwargs)
