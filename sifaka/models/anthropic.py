"""
Anthropic model implementation for Sifaka.

This module provides an implementation of the Model protocol for Anthropic models.
"""

import os
from typing import Optional, Any

try:
    import anthropic
    from anthropic import Anthropic, APIError, RateLimitError, APIConnectionError

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from sifaka.errors import ModelError, ModelAPIError, ConfigurationError
from sifaka.registry import register_model


class AnthropicModel:
    """Anthropic model implementation.

    This class implements the Model protocol for Anthropic models.

    Attributes:
        model_name: The name of the Anthropic model to use.
        api_key: The Anthropic API key to use. If not provided, it will be read from the
            ANTHROPIC_API_KEY environment variable.
        client: The Anthropic client instance.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None, **options: Any):
        """Initialize the Anthropic model.

        Args:
            model_name: The name of the Anthropic model to use.
            api_key: The Anthropic API key to use. If not provided, it will be read from the
                ANTHROPIC_API_KEY environment variable.
            **options: Additional options to pass to the Anthropic client.

        Raises:
            ConfigurationError: If the Anthropic package is not installed.
            ModelError: If the API key is not provided and not available in the environment.
        """
        if not ANTHROPIC_AVAILABLE:
            raise ConfigurationError(
                "Anthropic package not installed. Install it with 'pip install anthropic'."
            )

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.options = options

        if not self.api_key:
            raise ModelError(
                "Anthropic API key not provided. Either pass it as an argument or "
                "set the ANTHROPIC_API_KEY environment variable."
            )

        self.client = Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the Anthropic API.
                Supported options include:
                - temperature: Controls randomness. Higher values (e.g., 0.8) make output more random,
                  lower values (e.g., 0.2) make it more deterministic.
                - max_tokens: Maximum number of tokens to generate.
                - top_p: Controls diversity via nucleus sampling.
                - top_k: Controls diversity by limiting to top k tokens.
                - stop_sequences: Sequences where the API will stop generating further tokens.

        Returns:
            The generated text.

        Raises:
            ModelAPIError: If there is an error communicating with the Anthropic API.
        """
        import logging
        import time
        from sifaka.utils.error_handling import model_context, log_error

        logger = logging.getLogger(__name__)

        # Merge default options with provided options
        merged_options = {**self.options, **options}

        # Convert max_tokens to max_tokens_to_sample if present
        if "max_tokens" in merged_options:
            merged_options["max_tokens_to_sample"] = merged_options.pop("max_tokens")

        # Convert stop to stop_sequences if present
        if "stop" in merged_options:
            merged_options["stop_sequences"] = merged_options.pop("stop")

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
                    "max_tokens": merged_options.get("max_tokens_to_sample"),
                },
            ):
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **merged_options,
                )

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Log successful generation
                logger.debug(
                    f"Successfully generated text with Anthropic model '{self.model_name}' "
                    f"in {processing_time:.2f}ms, result length={len(response.content[0].text)}"
                )

                return response.content[0].text

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

        except APIConnectionError as e:
            # Log the error
            log_error(e, logger, component="AnthropicModel", operation="generation")

            # Raise as ModelAPIError with more context
            raise ModelAPIError(
                message=f"Error connecting to Anthropic API: {str(e)}",
                model_name=self.model_name,
                component="AnthropicModel",
                operation="generation",
                suggestions=[
                    "Check your internet connection",
                    "Verify that the Anthropic API is not experiencing an outage",
                    "Check if your firewall or proxy is blocking the connection",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": "APIConnectionError",
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

            # Raise as ModelAPIError with more context
            raise ModelAPIError(
                message=f"Unexpected error when calling Anthropic API: {str(e)}",
                model_name=self.model_name,
                component="AnthropicModel",
                operation="generation",
                suggestions=[
                    "Check the error message for details",
                    "Verify that your request is properly formatted",
                    "Check if there are any issues with the Anthropic service",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": type(e).__name__,
                    "prompt_length": len(prompt),
                },
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: The text to count tokens in.

        Returns:
            The number of tokens in the text.

        Raises:
            ModelError: If there is an error counting tokens.
        """
        import logging
        import time
        from sifaka.utils.error_handling import model_context, log_error

        logger = logging.getLogger(__name__)

        # Log token counting attempt
        logger.debug(
            f"Counting tokens for text of length {len(text)} with model '{self.model_name}'"
        )

        start_time = time.time()

        try:
            # Use model_context for consistent error handling
            with model_context(
                model_name=self.model_name,
                operation="token_counting",
                message_prefix="Failed to count tokens",
                suggestions=[
                    "Check if the Anthropic API is functioning properly",
                    "Verify that your API key is valid",
                ],
                metadata={"model_name": self.model_name, "text_length": len(text)},
            ):
                # Use Anthropic's token counting function
                token_count = self.client.count_tokens(text)

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Log successful token counting
                logger.debug(
                    f"Successfully counted {token_count} tokens for text of length {len(text)} "
                    f"in {processing_time:.2f}ms"
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
                    "Check if the Anthropic API is functioning properly",
                    "Verify that your API key is valid",
                    "Try with a shorter text",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": type(e).__name__,
                    "text_length": len(text),
                },
            )


@register_model("anthropic")
def create_anthropic_model(model_name: str, **options: Any) -> AnthropicModel:
    """Create an Anthropic model instance.

    This factory function creates an Anthropic model instance with the specified
    model name and options. It is registered with the registry system for
    dependency injection.

    Args:
        model_name: The name of the Anthropic model to use.
        **options: Additional options to pass to the Anthropic model constructor.

    Returns:
        An Anthropic model instance.

    Raises:
        ConfigurationError: If the Anthropic package is not installed.
        ModelError: If the API key is not provided and not available in the environment.
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

    except ConfigurationError as e:
        # Log the error
        log_error(e, logger, component="AnthropicModelFactory", operation="create_model")

        # Re-raise the error with more context
        raise ConfigurationError(
            message=f"Failed to create Anthropic model: {str(e)}",
            component="AnthropicModelFactory",
            operation="create_model",
            suggestions=["Install the Anthropic package with 'pip install anthropic'"],
            metadata={"model_name": model_name, "error_type": "ConfigurationError"},
        )

    except ModelError as e:
        # Log the error
        log_error(e, logger, component="AnthropicModelFactory", operation="create_model")

        # Re-raise the error with more context
        raise ModelError(
            message=f"Failed to create Anthropic model: {str(e)}",
            component="AnthropicModelFactory",
            operation="create_model",
            suggestions=[
                "Set the ANTHROPIC_API_KEY environment variable",
                "Pass the API key explicitly as api_key='your-api-key'",
            ],
            metadata={"model_name": model_name, "error_type": "ModelError"},
        )

    except Exception as e:
        # Log the error
        log_error(e, logger, component="AnthropicModelFactory", operation="create_model")

        # Raise as ModelError with more context
        raise ModelError(
            message=f"Unexpected error creating Anthropic model: {str(e)}",
            component="AnthropicModelFactory",
            operation="create_model",
            suggestions=[
                "Check the error message for details",
                "Verify that your options are valid",
            ],
            metadata={"model_name": model_name, "error_type": type(e).__name__},
        )
