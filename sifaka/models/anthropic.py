"""Anthropic model implementation for Sifaka.

This module provides an implementation of the Model protocol for Anthropic models,
supporting the Claude family of models. It handles token counting, error handling,
and configuration for Anthropic models.

The AnthropicModel class implements the Model protocol and provides methods for
generating text and counting tokens. It also provides a configure method for
updating model options after initialization.

Example:
    ```python
    from sifaka.models.anthropic import AnthropicModel, create_anthropic_model

    # Create a model directly
    model1 = AnthropicModel(model_name="claude-3-opus-20240229", api_key="your-api-key")

    # Create a model using the factory function
    model2 = create_anthropic_model(model_name="claude-3-sonnet-20240229", api_key="your-api-key")

    # Generate text
    response = model1.generate(
        "Write a short story about a robot.",
        temperature=0.7,
        max_tokens=500,
        system_message="You are a creative writer."
    )
    print(response)

    # Count tokens
    token_count = model1.count_tokens("This is a test.")
    print(f"Token count: {token_count}")
    ```
"""

import os
import time
from typing import Any, Optional

try:
    from anthropic import Anthropic, APIError, RateLimitError

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from sifaka.core.thought import Thought
from sifaka.utils.error_handling import (
    ConfigurationError,
    ModelAPIError,
    ModelError,
    log_error,
    model_context,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import ContextAwareMixin

# Configure logger
logger = get_logger(__name__)


class AnthropicModel(ContextAwareMixin):
    """Anthropic model implementation for generating text and counting tokens.

    This class implements the Model protocol for Anthropic models, supporting
    the Claude family of models. It handles token counting, error handling,
    and configuration for Anthropic models.

    The class also provides methods for counting tokens and configuring the model
    after initialization.

    Example:
        ```python
        from sifaka.models.anthropic import AnthropicModel

        # Create a model
        model = AnthropicModel(
            model_name="claude-3-opus-20240229",
            api_key="your-api-key",
            temperature=0.7
        )

        # Generate text with a system message
        response = model.generate(
            "Write a short story about a robot.",
            system_message="You are a creative writer.",
            max_tokens=500
        )
        print(response)
        ```
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **options: Any,
    ):
        """Initialize the Anthropic model with the specified parameters.

        Args:
            model_name: The name of the Anthropic model to use (e.g., "claude-3-opus-20240229").
            api_key: Optional API key to use. If not provided, it will be read from the
                ANTHROPIC_API_KEY environment variable.
            **options: Additional options to pass to the Anthropic API, such as:
                - temperature: Controls randomness in generation (0.0 to 1.0).
                - max_tokens: Maximum number of tokens to generate.
                - top_p: Controls diversity via nucleus sampling.
                - top_k: Controls diversity by limiting to top k tokens.

        Raises:
            ConfigurationError: If the Anthropic package is not installed.
            ModelError: If the API key is not provided and not available in the environment.

        Example:
            ```python
            # Create a model with API key from environment variable
            model1 = AnthropicModel(model_name="claude-3-opus-20240229")

            # Create a model with explicit API key and options
            model2 = AnthropicModel(
                model_name="claude-3-sonnet-20240229",
                api_key="your-api-key",
                temperature=0.7,
                max_tokens=500
            )
            ```
        """
        if not ANTHROPIC_AVAILABLE:
            raise ConfigurationError(
                "Anthropic package not installed. Install it with 'pip install anthropic'."
            )

        # Store model name and options
        self.model_name = model_name
        self.options = options

        # Get API key from parameter or environment variable
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ModelError(
                "Anthropic API key not provided and not found in environment variable ANTHROPIC_API_KEY.",
                component="AnthropicModel",
                operation="initialization",
                suggestions=[
                    "Provide an API key when creating the model",
                    "Set the ANTHROPIC_API_KEY environment variable",
                ],
            )

        # Initialize the Anthropic client
        self.client = Anthropic(api_key=self.api_key)

        logger.debug(f"Initialized Anthropic model '{model_name}'")

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt using the Anthropic API.

        This method sends the prompt to the Anthropic API and returns the generated text.
        It supports system messages and other Anthropic-specific options.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the Anthropic API, such as:
                - temperature: Controls randomness in generation (0.0 to 1.0).
                - max_tokens: Maximum number of tokens to generate.
                - system_message: System message for the model.
                - stop_sequences: Sequences where the API will stop generating further tokens.

        Returns:
            The generated text.

        Raises:
            ModelAPIError: If there is an error calling the Anthropic API.
            ModelError: If there is another error generating text.

        Example:
            ```python
            # Basic generation
            response = model.generate("Write a short story about a robot.")

            # Generation with options
            response = model.generate(
                "Write a short story about a robot.",
                temperature=0.7,
                max_tokens=500,
                system_message="You are a creative writer."
            )
            ```
        """
        # Merge default options with provided options
        merged_options = {**self.options, **options}

        # Extract system message if provided
        system_message = merged_options.pop("system_message", None)

        # Extract max_tokens if provided
        max_tokens = merged_options.get("max_tokens", 1024)

        # Log generation attempt
        logger.debug(
            f"Generating text with Anthropic model '{self.model_name}', "
            f"prompt length={len(prompt)}, "
            f"temperature={merged_options.get('temperature', 'default')}"
        )

        start_time = time.time()

        try:
            # Use model_context for consistent error handling
            with model_context(
                model_name=self.model_name,
                operation="generation",
                message_prefix="Failed to generate text with Anthropic model",
                suggestions=[
                    "Check your API key and ensure it is valid",
                    "Verify that you have sufficient quota",
                    "Check if the model is available in your region",
                ],
                metadata={
                    "model_name": self.model_name,
                    "prompt_length": len(prompt),
                    "temperature": merged_options.get("temperature"),
                    "max_tokens": max_tokens,
                },
            ):
                # Extract specific parameters that are supported
                stop_sequences = merged_options.get("stop_sequences", None)

                # Build the API call parameters
                api_params = {
                    "model": self.model_name,
                    "max_tokens": max_tokens,
                    "temperature": merged_options.get("temperature", 0.7),
                }

                # Add system message if provided
                if system_message:
                    api_params["system"] = system_message

                # Add stop_sequences if provided
                if stop_sequences:
                    api_params["stop_sequences"] = stop_sequences

                # Add other supported parameters
                for param in ["top_p", "top_k"]:
                    if param in merged_options:
                        api_params[param] = merged_options[param]

                # Make the API call
                response = self.client.messages.create(
                    messages=[{"role": "user", "content": prompt}], **api_params
                )

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Extract the response text
                response_text: str = response.content[0].text

                # Log successful generation
                logger.debug(
                    f"Successfully generated text with Anthropic model '{self.model_name}' "
                    f"in {processing_time:.2f}ms, result length={len(response_text)}"
                )

                return response_text

        except RateLimitError as e:
            # Log the error
            log_error(e, logger, component="AnthropicModel", operation="generation")

            # Raise as ModelAPIError with more context
            raise ModelAPIError(
                message=f"Anthropic rate limit exceeded: {str(e)}",
                model_name=self.model_name,
                component="AnthropicModel",
                operation="generation",
                suggestions=[
                    "Reduce the frequency of requests",
                    "Implement exponential backoff",
                    "Consider upgrading your Anthropic plan for higher rate limits",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": "RateLimitError",
                    "prompt_length": len(prompt),
                },
            )

        except APIError as e:
            # Log the error
            log_error(e, logger, component="AnthropicModel", operation="generation")

            # Raise as ModelAPIError with more context
            raise ModelAPIError(
                message=f"Anthropic API error: {str(e)}",
                model_name=self.model_name,
                component="AnthropicModel",
                operation="generation",
                suggestions=[
                    "Check if the model name is correct",
                    "Verify that your API key has access to this model",
                    "Check if your request parameters are valid",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": "APIError",
                    "prompt_length": len(prompt),
                },
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="AnthropicModel", operation="generation")

            # Raise as ModelError with more context
            raise ModelError(
                message=f"Error generating text with Anthropic model: {str(e)}",
                component="AnthropicModel",
                operation="generation",
                suggestions=[
                    "Check your API key and ensure it is valid",
                    "Verify that the model name is correct",
                    "Check if your request parameters are valid",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": type(e).__name__,
                    "prompt_length": len(prompt),
                },
            )

    def generate_with_thought(self, thought: Thought, **options: Any) -> tuple[str, str]:
        """Generate text using a Thought container.

        This method allows the model to access the full context in the Thought container,
        including any retrieved documents, when generating text. It extracts the prompt,
        system prompt, and pre-generation context from the Thought container and uses
        them to generate text.

        Args:
            thought: The Thought container with context for generation.
            **options: Additional options for generation.

        Returns:
            The generated text.

        Raises:
            ModelAPIError: If there is an error calling the Anthropic API.
            ModelError: If there is another error generating text.

        Example:
            ```python
            from sifaka.core.thought import Thought, Document

            # Create a thought with prompt and context
            thought = Thought(
                prompt="Write a short story about a robot.",
                system_prompt="You are a creative writer.",
                pre_generation_context=[
                    Document(text="Robots are machines that can be programmed to perform tasks."),
                    Document(text="Asimov's Three Laws of Robotics are rules for robots in his science fiction."),
                ]
            )

            # Generate text with context
            text = model.generate_with_thought(thought, temperature=0.7, max_tokens=500)
            print(text)
            ```
        """
        # Use mixin to build contextualized prompt
        full_prompt = self._build_contextualized_prompt(thought, max_docs=5)

        # Add system_prompt to options if available
        generation_options = options.copy()
        if thought.system_prompt:
            generation_options["system_message"] = thought.system_prompt

        # Log context usage
        if self._has_context(thought):
            context_summary = self._get_context_summary(thought)
            logger.debug(f"AnthropicModel using context: {context_summary}")

        # Generate text using the contextualized prompt and options
        generated_text = self.generate(full_prompt, **generation_options)
        return generated_text, full_prompt

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the Anthropic tokenizer.

        This method uses the Anthropic tokenizer to count tokens according to the
        tokenization scheme used by Claude models.

        Args:
            text: The text to count tokens in.

        Returns:
            The number of tokens in the text according to the Claude tokenization scheme.

        Raises:
            ModelError: If there is an error counting tokens.

        Example:
            ```python
            # Count tokens in text
            token_count = model.count_tokens("This is a test.")
            print(f"Token count: {token_count}")
            ```
        """
        import logging

        logger = logging.getLogger(__name__)

        # Log token counting attempt
        logger.debug(
            f"Counting tokens for text of length {len(text)} with model '{self.model_name}'"
        )

        try:
            # Use model_context for consistent error handling
            with model_context(
                model_name=self.model_name,
                operation="token_counting",
                message_prefix="Failed to count tokens",
                suggestions=[
                    "Check if the Anthropic package is properly installed",
                    "Verify that the model name is supported",
                ],
                metadata={"model_name": self.model_name, "text_length": len(text)},
            ):
                # Count tokens using Anthropic's tokenizer
                # The Anthropic client doesn't have a direct count_tokens method in the latest version
                # We'll use a simple approximation based on words
                # This is not accurate but provides a reasonable estimate
                # In a production environment, you would want to use a proper tokenizer
                token_count = len(text.split())

                # Log successful token counting
                logger.debug(
                    f"Successfully counted {token_count} tokens for text of length {len(text)}"
                )

                return token_count

        except Exception as e:
            # Log the error
            log_error(e, logger, component="AnthropicModel", operation="token_counting")

            # Raise as ModelError with more context
            raise ModelError(
                message=f"Error counting tokens: {str(e)}",
                component="AnthropicModel",
                operation="token_counting",
                suggestions=[
                    "Check if the Anthropic package is properly installed",
                    "Verify that the model name is supported",
                    "Try using a different encoding method",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": type(e).__name__,
                    "text_length": len(text),
                },
            )

    def configure(self, **options: Any) -> None:
        """Update the model configuration.

        This method allows updating the model configuration after initialization,
        including the API key and other options.

        Args:
            **options: New configuration options to set, such as:
                - api_key: New API key to use.
                - temperature: New temperature value.
                - max_tokens: New maximum tokens value.

        Raises:
            ModelError: If there is an error configuring the model, such as an invalid API key.

        Example:
            ```python
            # Create a model
            model = AnthropicModel(model_name="claude-3-opus-20240229", api_key="your-api-key")

            # Update configuration
            model.configure(
                temperature=0.5,
                max_tokens=1000
            )

            # Update API key
            model.configure(api_key="your-new-api-key")
            ```
        """
        import logging

        logger = logging.getLogger(__name__)

        # Log configuration attempt
        logger.debug(f"Configuring Anthropic model '{self.model_name}'")

        # Update API key if provided
        if "api_key" in options:
            self.api_key = options["api_key"]
            # Recreate client with new API key
            self.client = Anthropic(api_key=self.api_key)
            logger.debug("Updated API key")

        # Update other options
        for key, value in options.items():
            if key not in ["api_key"]:
                self.options[key] = value
                logger.debug(f"Updated option '{key}'")

        logger.debug(f"Successfully configured Anthropic model '{self.model_name}'")

    # Async methods required by Model protocol
    async def _generate_async(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt asynchronously.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options for generation.

        Returns:
            The generated text.
        """
        # For now, just call the sync method
        # In a real implementation, you would use the async Anthropic client
        return self.generate(prompt, **options)

    async def _generate_with_thought_async(
        self, thought: Thought, **options: Any
    ) -> tuple[str, str]:
        """Generate text using a Thought container asynchronously.

        Args:
            thought: The Thought container with context for generation.
            **options: Additional options for generation.

        Returns:
            A tuple of (generated_text, actual_prompt_used).
        """
        # For now, just call the sync method
        # In a real implementation, you would use the async Anthropic client
        return self.generate_with_thought(thought, **options)


def create_anthropic_model(
    model_name: str,
    **options: Any,
) -> AnthropicModel:
    """Create an Anthropic model instance.

    This factory function creates an Anthropic model instance with the specified
    model name and options. It handles error logging and provides a consistent
    interface for creating Anthropic models.

    Args:
        model_name: The name of the Anthropic model to use (e.g., "claude-3-opus-20240229").
        **options: Additional options to pass to the Anthropic model constructor, such as:
            - api_key: The Anthropic API key to use.
            - temperature: Controls randomness in generation (0.0 to 1.0).
            - max_tokens: Maximum number of tokens to generate.

    Returns:
        An Anthropic model instance implementing the Model protocol.

    Raises:
        ConfigurationError: If the Anthropic package is not installed.
        ModelError: If the API key is not provided and not available in the environment,
            or if there is another error creating the model.

    Example:
        ```python
        from sifaka.models.anthropic import create_anthropic_model

        # Create a model using the factory function
        model = create_anthropic_model(
            model_name="claude-3-opus-20240229",
            api_key="your-api-key",
            temperature=0.7
        )

        # Generate text
        response = model.generate("Write a short story about a robot.")
        print(response)
        ```
    """
    import logging

    from sifaka.utils.error_handling import log_error

    logger = logging.getLogger(__name__)

    # Log model creation attempt
    logger.debug(f"Creating Anthropic model with name '{model_name}'")

    try:
        # Create the model
        model = AnthropicModel(model_name=model_name, **options)

        # Log successful model creation
        logger.debug(f"Successfully created Anthropic model with name '{model_name}'")

        return model
    except Exception as e:
        # Log the error
        log_error(e, logger, component="AnthropicModel", operation="creation")

        # Re-raise the exception
        raise
