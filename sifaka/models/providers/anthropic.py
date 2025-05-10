"""
Anthropic Model Provider

This module provides the AnthropicProvider class which implements the ModelProviderCore
interface for Anthropic Claude models, and additional Anthropic-specific functionality
like text reflection and analysis.

## Overview
The Anthropic provider connects to Anthropic's API for text generation, offering access
to Claude models like Claude 3 Opus, Claude 3 Sonnet, and others. It handles authentication,
API communication, token counting, response processing, and specialized text analysis.

## Components
- **AnthropicProvider**: Main provider class for Anthropic models
- **AnthropicClient**: API client implementation for Anthropic
- **AnthropicTokenCounter**: Token counter implementation for Anthropic models
- **AnthropicReflector**: Specialized component for text analysis and reflection

## Usage Example
```python
from sifaka.models.providers.anthropic import AnthropicProvider
from sifaka.models.base import ModelConfig

# Create a provider with default settings
provider = AnthropicProvider(model_name="claude-3-opus-20240229")

# Generate text
response = provider.generate("Explain quantum computing in simple terms.")

# Create a provider with custom configuration
config = ModelConfig().with_temperature(0.9).with_max_tokens(2000)
provider = AnthropicProvider(
    model_name="claude-3-sonnet-20240229",
    config=config
)

# Use the reflector for text analysis
reflector = provider.create_reflector()
analysis = reflector.reflect("This is a sample text to analyze.")
```
"""

import time
import os
from typing import Optional, Dict, Any, ClassVar, List, Union

import anthropic
import tiktoken
from anthropic import Anthropic
from pydantic import BaseModel, Field

from sifaka.models.base import APIClient, ModelConfig, TokenCounter
from sifaka.models.core import ModelProviderCore
from sifaka.utils.logging import get_logger
from sifaka.utils.tracing import Tracer

logger = get_logger(__name__)


# ReflectionResult class from integrations/anthropic.py
class ReflectionResult(BaseModel):
    """
    Result of a text reflection operation.

    This class represents the result of analyzing text using Anthropic's API.
    It includes the original text, analysis results, suggestions for improvement,
    and safety scoring.

    Attributes:
        text: The original text that was analyzed
        analysis: Dictionary containing analysis results
        suggestions: Optional dictionary of suggestions for improvement
        safety_score: Optional safety score (0-1) indicating potential issues
        processing_time_ms: Time taken to perform the analysis in milliseconds

    Examples:
        ```python
        from sifaka.models.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(model_name="claude-3-opus-20240229")
        reflector = provider.create_reflector()
        result = reflector.reflect("This is a sample text to analyze.")

        print(f"Analysis: {result.analysis}")
        if result.suggestions:
            print(f"Suggestions: {result.suggestions}")
        ```
    """

    text: str = Field(description="The original text that was analyzed")
    analysis: Dict[str, Any] = Field(description="Dictionary containing analysis results")
    suggestions: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional dictionary of suggestions for improvement"
    )
    safety_score: Optional[float] = Field(
        default=None, description="Optional safety score (0-1) indicating potential issues"
    )
    processing_time_ms: Optional[float] = Field(
        default=None, description="Time taken to perform the analysis in milliseconds"
    )


class AnthropicClient(APIClient):
    """
    Anthropic API client implementation.

    This client handles communication with Anthropic's API for Claude models.
    It manages authentication, request formatting, and response processing.

    Lifecycle:
        1. Initialization: Set up with API key
        2. Request Preparation: Format prompts for the API
        3. API Communication: Send requests to Anthropic
        4. Response Processing: Parse and return responses
        5. Error Handling: Handle API errors and retries

    Examples:
        ```python
        from sifaka.models.providers.anthropic import AnthropicClient
        from sifaka.models.base import ModelConfig

        # Create client with API key
        client = AnthropicClient(api_key="sk-ant-api...")

        # Create configuration
        config = ModelConfig(temperature=0.7, max_tokens=1000)

        # Send prompt
        response = client.send_prompt("Explain quantum computing", config)
        ```
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key (if None, will try to get from environment)
        """
        # Check for API key in environment if not provided
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                logger.debug(f"Retrieved API key from environment: {api_key[:10]}...")
            else:
                logger.warning(
                    "No Anthropic API key provided and ANTHROPIC_API_KEY environment variable not set"
                )

        # Validate API key format
        if api_key and not api_key.startswith("sk-ant-api"):
            logger.warning(
                f"API key format appears incorrect. Expected to start with 'sk-ant-api', got: {api_key[:10]}..."
            )

        # Initialize client
        try:
            self.client = Anthropic(api_key=api_key)
            logger.debug("Initialized Anthropic client")
            self._api_key = api_key
            self._request_count = 0
            self._error_count = 0
            self._last_request_time = None
            self._last_response_time = None
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {e}")
            raise ValueError(f"Failed to initialize Anthropic client: {str(e)}")

    def send_prompt(self, prompt: str, config: ModelConfig) -> str:
        """
        Send a prompt to Anthropic and return the response.

        Args:
            prompt: The prompt to send
            config: Configuration for the request

        Returns:
            The generated text response

        Raises:
            ValueError: If no API key is provided
            RuntimeError: If the API request fails
        """
        start_time = time.time()
        self._last_request_time = start_time

        try:
            # Get API key from config or client
            api_key = config.api_key or self._api_key

            # Check for missing API key
            if not api_key:
                raise ValueError(
                    "No API key provided. Please provide an API key either by setting the "
                    "ANTHROPIC_API_KEY environment variable or by passing it explicitly."
                )

            # Validate input
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt must be a non-empty string")

            # Send request to API
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            # Update statistics
            self._request_count += 1
            self._last_response_time = time.time()

            # Return response text
            return response.content[0].text

        except anthropic.AnthropicError as e:
            self._error_count += 1
            logger.error(f"Anthropic API error: {str(e)}")
            raise RuntimeError(f"Anthropic API error: {str(e)}")

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error sending prompt to Anthropic: {e}")
            raise RuntimeError(f"Error sending prompt to Anthropic: {str(e)}")

        finally:
            # Log request duration
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            logger.debug(f"Anthropic request completed in {duration_ms:.2f}ms")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get client usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "last_request_time": self._last_request_time,
            "last_response_time": self._last_response_time,
        }


class AnthropicTokenCounter(TokenCounter):
    """
    Token counter using tiktoken for Anthropic models.

    This class provides token counting functionality for Anthropic Claude models
    using the tiktoken library. It uses the cl100k_base encoding which is
    compatible with Claude models.

    Lifecycle:
        1. Initialization: Set up with model name
        2. Token Counting: Count tokens in text
        3. Statistics: Track usage statistics

    Examples:
        ```python
        from sifaka.models.providers.anthropic import AnthropicTokenCounter

        # Create token counter
        counter = AnthropicTokenCounter(model="claude-3-opus-20240229")

        # Count tokens in text
        token_count = counter.count_tokens("This is a test.")
        print(f"Token count: {token_count}")
        ```
    """

    def __init__(self, model: str = "claude-3-opus-20240229") -> None:
        """
        Initialize the token counter for a specific model.

        Args:
            model: The model name to use for token counting
        """
        try:
            # Anthropic uses cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.model = model
            logger.debug(f"Initialized token counter for model {model}")

            # Initialize statistics
            self._count_calls = 0
            self._total_tokens_counted = 0
            self._error_count = 0
            self._last_count_time = None

        except Exception as e:
            logger.error(f"Error initializing token counter: {str(e)}")
            raise ValueError(f"Failed to initialize token counter: {str(e)}")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the text using the model's encoding.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text

        Raises:
            ValueError: If text is not a string
            RuntimeError: If token counting fails
        """
        start_time = time.time()
        self._last_count_time = start_time

        try:
            # Validate input
            if not isinstance(text, str):
                raise ValueError("Text must be a string")

            # Count tokens
            token_count = len(self.encoding.encode(text))

            # Update statistics
            self._count_calls += 1
            self._total_tokens_counted += token_count

            return token_count

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error counting tokens: {str(e)}")
            raise RuntimeError(f"Error counting tokens: {str(e)}")

        finally:
            # Log duration
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            logger.debug(f"Token counting completed in {duration_ms:.2f}ms")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get token counter usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "model": self.model,
            "count_calls": self._count_calls,
            "total_tokens_counted": self._total_tokens_counted,
            "error_count": self._error_count,
            "last_count_time": self._last_count_time,
            "average_tokens_per_call": (
                self._total_tokens_counted / self._count_calls if self._count_calls > 0 else 0
            ),
        }


class AnthropicReflector:
    """
    Reflector that uses Anthropic's API for text analysis.

    This class provides text analysis capabilities using Anthropic's Claude models.
    It can analyze content, style, tone, and provide suggestions for improvement.

    Lifecycle:
        1. Initialization: Set up with API key and model
        2. Reflection: Analyze text content
        3. Result Processing: Parse and structure analysis results
        4. Statistics: Track usage and performance

    Examples:
        ```python
        from sifaka.models.providers.anthropic import AnthropicProvider, AnthropicReflector

        # Create reflector directly
        reflector = AnthropicReflector(
            api_key="sk-ant-api...",
            model="claude-3-opus-20240229"
        )

        # Or create through provider
        provider = AnthropicProvider(model_name="claude-3-opus-20240229")
        reflector = provider.create_reflector()

        # Analyze text
        result = reflector.reflect("This is a sample text to analyze.")
        print(f"Analysis: {result.analysis}")
        ```
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize the Anthropic reflector.

        Args:
            api_key: Anthropic API key
            model: Model to use for reflection
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        """
        try:
            self.client = Anthropic(api_key=api_key)
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens

            # Initialize statistics
            self._reflection_count = 0
            self._error_count = 0
            self._total_processing_time = 0
            self._last_reflection_time = None

            logger.debug(f"Initialized Anthropic reflector with model {model}")

        except Exception as e:
            logger.error(f"Error initializing Anthropic reflector: {e}")
            raise ValueError(f"Failed to initialize Anthropic reflector: {str(e)}")

    def reflect(self, text: str) -> ReflectionResult:
        """
        Reflect on the given text using Anthropic's API.

        Args:
            text: Text to reflect on

        Returns:
            ReflectionResult containing analysis and suggestions

        Raises:
            ValueError: If text is not a string or is empty
            RuntimeError: If reflection fails
        """
        start_time = time.time()
        self._last_reflection_time = start_time

        try:
            # Validate input
            if not isinstance(text, str):
                raise ValueError("Text must be a string")

            if not text.strip():
                raise ValueError("Text cannot be empty")

            # Prepare the prompt
            prompt = f"""Please analyze the following text and provide:
1. A detailed analysis of its content, style, and tone
2. Suggestions for improvement
3. A safety score (0-1) indicating potential issues

Text to analyze:
{text}

Please provide your analysis in a structured format with clear sections for:
- Content Analysis
- Style Analysis
- Tone Analysis
- Improvement Suggestions
- Safety Assessment (with a score from 0 to 1)"""

            # Call the API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse the response
            response_text = response.content[0].text

            # Extract suggestions (simple approach - could be improved with more structured parsing)
            suggestions = {}
            if "Improvement Suggestions" in response_text:
                suggestions_section = response_text.split("Improvement Suggestions")[1].split(
                    "Safety Assessment"
                )[0]
                suggestions = {
                    "content": suggestions_section.strip(),
                    "items": [
                        s.strip() for s in suggestions_section.strip().split("\n") if s.strip()
                    ],
                }

            # Extract safety score (simple approach)
            safety_score = 0.8  # Default placeholder
            if "Safety Assessment" in response_text:
                safety_section = response_text.split("Safety Assessment")[1]
                # Try to find a number between 0 and 1
                import re

                score_matches = re.findall(
                    r"(\d+\.\d+|\d+)/1|score:?\s*(\d+\.\d+|\d+)", safety_section.lower()
                )
                if score_matches:
                    # Use the first match
                    match = score_matches[0]
                    score_str = match[0] if match[0] else match[1]
                    try:
                        safety_score = float(score_str)
                        # Ensure it's between 0 and 1
                        safety_score = min(max(safety_score, 0), 1)
                    except ValueError:
                        pass

            # Create analysis dictionary
            analysis = {
                "content": response_text,
                "model": self.model,
                "temperature": self.temperature,
                "sections": {
                    "content": (
                        response_text.split("Content Analysis")[1]
                        .split("Style Analysis")[0]
                        .strip()
                        if "Content Analysis" in response_text and "Style Analysis" in response_text
                        else ""
                    ),
                    "style": (
                        response_text.split("Style Analysis")[1].split("Tone Analysis")[0].strip()
                        if "Style Analysis" in response_text and "Tone Analysis" in response_text
                        else ""
                    ),
                    "tone": (
                        response_text.split("Tone Analysis")[1]
                        .split("Improvement Suggestions")[0]
                        .strip()
                        if "Tone Analysis" in response_text
                        and "Improvement Suggestions" in response_text
                        else ""
                    ),
                },
            }

            # Calculate processing time
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000

            # Update statistics
            self._reflection_count += 1
            self._total_processing_time += processing_time_ms

            # Create and return result
            return ReflectionResult(
                text=text,
                analysis=analysis,
                suggestions=suggestions,
                safety_score=safety_score,
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error in Anthropic reflection: {str(e)}")
            raise RuntimeError(f"Error in Anthropic reflection: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get reflector usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "model": self.model,
            "reflection_count": self._reflection_count,
            "error_count": self._error_count,
            "total_processing_time": self._total_processing_time,
            "last_reflection_time": self._last_reflection_time,
            "average_processing_time": (
                self._total_processing_time / self._reflection_count
                if self._reflection_count > 0
                else 0
            ),
        }


class AnthropicProvider(ModelProviderCore):
    """
    Anthropic model provider implementation.

    This provider supports Claude models with configurable parameters,
    built-in token counting, and specialized text analysis capabilities.
    It handles communication with Anthropic's API, token counting, and
    response processing.

    Lifecycle:
        1. Initialization: Set up with model name and configuration
        2. Client Creation: Create API client and token counter
        3. Text Generation: Generate text from prompts
        4. Token Counting: Count tokens for optimization
        5. Reflection: Analyze text content
        6. Statistics: Track usage and performance

    Examples:
        ```python
        from sifaka.models.providers.anthropic import AnthropicProvider
        from sifaka.models.base import ModelConfig

        # Create a provider with default settings
        provider = AnthropicProvider(model_name="claude-3-opus-20240229")

        # Generate text
        response = provider.generate("Explain quantum computing in simple terms.")

        # Count tokens
        token_count = provider.count_tokens("How many tokens is this?")

        # Create a provider with custom configuration
        config = ModelConfig().with_temperature(0.9).with_max_tokens(2000)
        provider = AnthropicProvider(
            model_name="claude-3-sonnet-20240229",
            config=config
        )

        # Use the reflector for text analysis
        reflector = provider.create_reflector()
        analysis = reflector.reflect("This is a sample text to analyze.")
        ```
    """

    # Class constants
    DEFAULT_MODEL: ClassVar[str] = "claude-3-opus-20240229"
    AVAILABLE_MODELS: ClassVar[List[str]] = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
        "claude-instant-1.2",
    ]

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        config: Optional[ModelConfig] = None,
        api_client: Optional[APIClient] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        """
        Initialize the Anthropic provider.

        Args:
            model_name: The name of the model to use
            config: Optional model configuration
            api_client: Optional API client to use
            token_counter: Optional token counter to use

        Raises:
            ImportError: If the Anthropic package is not installed
            ValueError: If the model name is not supported
        """
        # Verify Anthropic package is installed
        try:
            # Just importing the package to verify it's installed
            # We already imported it at the module level
            pass
        except ImportError:
            raise ImportError("Anthropic package is required. Install with: pip install anthropic")

        # Validate model name
        if model_name not in self.AVAILABLE_MODELS:
            logger.warning(
                f"Model '{model_name}' is not in the list of known Anthropic models. "
                f"Available models: {', '.join(self.AVAILABLE_MODELS)}"
            )

        # Initialize base class
        super().__init__(
            model_name=model_name,
            config=config,
            api_client=api_client,
            token_counter=token_counter,
        )

        # Initialize statistics
        self._generation_count = 0
        self._token_count_calls = 0
        self._reflection_count = 0
        self._error_count = 0
        self._total_processing_time = 0
        self._last_generation_time = None

        logger.debug(f"Initialized Anthropic provider with model {model_name}")

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the model with a prompt (delegates to generate).

        This method is needed for compatibility with the critique service
        which expects an 'invoke' method.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional keyword arguments to pass to the model

        Returns:
            The generated text response
        """
        start_time = time.time()
        self._last_generation_time = start_time

        try:
            result = self.generate(prompt, **kwargs)

            # Update statistics
            self._generation_count += 1
            self._total_processing_time += (time.time() - start_time) * 1000

            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error invoking model: {e}")
            raise

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """
        Asynchronously invoke the model with a prompt.

        This method delegates to agenerate if it exists, or falls back to
        synchronous generate.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional keyword arguments to pass to the model

        Returns:
            The generated text response
        """
        start_time = time.time()

        try:
            if hasattr(self, "agenerate"):
                result = await self.agenerate(prompt, **kwargs)
            else:
                # Fall back to synchronous generate
                result = self.generate(prompt, **kwargs)

            # Update statistics
            self._generation_count += 1
            self._total_processing_time += (time.time() - start_time) * 1000

            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error asynchronously invoking model: {e}")
            raise

    def _create_default_client(self) -> APIClient:
        """
        Create a default Anthropic client.

        Returns:
            An AnthropicClient instance
        """
        return AnthropicClient(api_key=self.config.api_key)

    def _create_default_token_counter(self) -> TokenCounter:
        """
        Create a default token counter for the current model.

        Returns:
            An AnthropicTokenCounter instance
        """
        return AnthropicTokenCounter(model=self.model_name)

    def create_reflector(
        self, temperature: float = 0.7, max_tokens: int = 1000
    ) -> AnthropicReflector:
        """
        Create a reflector for text analysis.

        Args:
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            An AnthropicReflector instance

        Examples:
            ```python
            from sifaka.models.providers.anthropic import AnthropicProvider

            # Create provider
            provider = AnthropicProvider(model_name="claude-3-opus-20240229")

            # Create reflector
            reflector = provider.create_reflector(
                temperature=0.5,
                max_tokens=2000
            )

            # Analyze text
            result = reflector.reflect("This is a sample text to analyze.")
            print(f"Analysis: {result.analysis}")
            ```
        """
        self._reflection_count += 1

        return AnthropicReflector(
            api_key=self.config.api_key,
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the text.

        This method overrides the base implementation to track statistics.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text
        """
        self._token_count_calls += 1
        return super().count_tokens(text)

    @property
    def name(self) -> str:
        """
        Get the provider name.

        Returns:
            The provider name
        """
        return f"Anthropic-{self.model_name}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get provider usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        # Get client statistics if available
        client_stats = {}
        if hasattr(self.client, "get_statistics") and callable(self.client.get_statistics):
            client_stats = self.client.get_statistics()

        # Get token counter statistics if available
        counter_stats = {}
        if hasattr(self.token_counter, "get_statistics") and callable(
            self.token_counter.get_statistics
        ):
            counter_stats = self.token_counter.get_statistics()

        return {
            "model": self.model_name,
            "generation_count": self._generation_count,
            "token_count_calls": self._token_count_calls,
            "reflection_count": self._reflection_count,
            "error_count": self._error_count,
            "total_processing_time": self._total_processing_time,
            "last_generation_time": self._last_generation_time,
            "average_processing_time": (
                self._total_processing_time / self._generation_count
                if self._generation_count > 0
                else 0
            ),
            "client": client_stats,
            "token_counter": counter_stats,
        }
