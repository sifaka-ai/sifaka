"""
OpenAI model implementation for Sifaka.

This module provides an implementation of the Model protocol for OpenAI models.
It also provides a factory function for creating OpenAI model instances.
"""

import os
from typing import Optional, Any

try:
    import tiktoken
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError

    # No need to import specific types for now

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from sifaka.errors import ModelError, ModelAPIError, ConfigurationError
from sifaka.registry import register_model


class OpenAIModel:
    """OpenAI model implementation.

    This class implements the Model protocol for OpenAI models.

    Attributes:
        model_name: The name of the OpenAI model to use.
        api_key: The OpenAI API key to use. If not provided, it will be read from the
            OPENAI_API_KEY environment variable.
        client: The OpenAI client instance.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        **options: Any,
    ):
        """Initialize the OpenAI model.

        Args:
            model_name: The name of the OpenAI model to use.
            api_key: The OpenAI API key to use. If not provided, it will be read from the
                OPENAI_API_KEY environment variable.
            organization: The OpenAI organization ID to use.
            **options: Additional options to pass to the OpenAI client.

        Raises:
            ConfigurationError: If the OpenAI package is not installed.
            ModelError: If the API key is not provided and not available in the environment.
        """
        if not OPENAI_AVAILABLE:
            raise ConfigurationError(
                "OpenAI package not installed. Install it with 'pip install openai'."
            )

        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.organization = organization
        self.options = options

        if not self.api_key:
            raise ModelError(
                "OpenAI API key not provided. Either pass it as an argument or "
                "set the OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.organization,
        )

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the OpenAI API.
                Supported options include:
                - temperature: Controls randomness. Higher values (e.g., 0.8) make output more random,
                  lower values (e.g., 0.2) make it more deterministic.
                - max_tokens: Maximum number of tokens to generate.
                - top_p: Controls diversity via nucleus sampling.
                - frequency_penalty: Reduces repetition of token sequences.
                - presence_penalty: Reduces repetition of topics.
                - stop: Sequences where the API will stop generating further tokens.
                - system_message: A system message to include at the beginning of the conversation.
                - use_completion_api: Force using the completion API instead of chat API.

        Returns:
            The generated text.

        Raises:
            ModelAPIError: If there is an error communicating with the OpenAI API.
        """
        import logging
        from sifaka.utils.error_handling import model_context, log_error

        logger = logging.getLogger(__name__)

        # Merge default options with provided options
        merged_options = {**self.options, **options}

        # Determine if we should use the completion API
        use_completion_api = merged_options.pop("use_completion_api", False)

        # Extract system message if provided
        system_message = merged_options.pop("system_message", None)

        # Convert stop_sequences to stop for OpenAI compatibility
        if "stop_sequences" in merged_options:
            merged_options["stop"] = merged_options.pop("stop_sequences")

        # Log generation attempt
        logger.debug(
            f"Generating text with OpenAI model '{self.model_name}', "
            f"prompt length={len(prompt)}, "
            f"temperature={merged_options.get('temperature', 'default')}"
        )

        try:
            # Use the appropriate API based on the model and options
            with model_context(
                model_name=self.model_name,
                operation="generation",
                message_prefix="Failed to generate text with OpenAI model",
                suggestions=[
                    "Check your API key and ensure it is valid",
                    "Verify that you have sufficient quota",
                    "Check if the model is available in your region",
                ],
                metadata={
                    "model_name": self.model_name,
                    "prompt_length": len(prompt),
                    "temperature": merged_options.get("temperature"),
                    "max_tokens": merged_options.get("max_tokens"),
                },
            ):
                if not use_completion_api and self._is_chat_model():
                    result = self._generate_chat(prompt, system_message, **merged_options)
                else:
                    result = self._generate_completion(prompt, **merged_options)

                # Log successful generation
                logger.debug(
                    f"Successfully generated text with OpenAI model '{self.model_name}', "
                    f"result length={len(result)}"
                )

                return result

        except RateLimitError as e:
            # Log the error
            log_error(e, logger, component="OpenAIModel", operation="generation")

            # Raise as ModelAPIError with more context
            raise ModelAPIError(
                message=f"OpenAI rate limit exceeded: {str(e)}",
                model_name=self.model_name,
                component="OpenAIModel",
                operation="generation",
                suggestions=[
                    "Reduce the frequency of requests",
                    "Implement exponential backoff",
                    "Consider upgrading your OpenAI plan for higher rate limits",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": "RateLimitError",
                    "prompt_length": len(prompt),
                },
            )

        except APIConnectionError as e:
            # Log the error
            log_error(e, logger, component="OpenAIModel", operation="generation")

            # Raise as ModelAPIError with more context
            raise ModelAPIError(
                message=f"Error connecting to OpenAI API: {str(e)}",
                model_name=self.model_name,
                component="OpenAIModel",
                operation="generation",
                suggestions=[
                    "Check your internet connection",
                    "Verify that the OpenAI API is not experiencing an outage",
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
            log_error(e, logger, component="OpenAIModel", operation="generation")

            # Raise as ModelAPIError with more context
            raise ModelAPIError(
                message=f"OpenAI API error: {str(e)}",
                model_name=self.model_name,
                component="OpenAIModel",
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
            log_error(e, logger, component="OpenAIModel", operation="generation")

            # Raise as ModelAPIError with more context
            raise ModelAPIError(
                message=f"Unexpected error when calling OpenAI API: {str(e)}",
                model_name=self.model_name,
                component="OpenAIModel",
                operation="generation",
                suggestions=[
                    "Check the error message for details",
                    "Verify that your request is properly formatted",
                    "Check if there are any issues with the OpenAI service",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": type(e).__name__,
                    "prompt_length": len(prompt),
                },
            )

    def _generate_chat(
        self, prompt: str, system_message: Optional[str] = None, **options: Any
    ) -> str:
        """Generate text using the chat completions API.

        Args:
            prompt: The prompt to generate text from.
            system_message: Optional system message to include.
            **options: Additional options to pass to the OpenAI API.

        Returns:
            The generated text.
        """
        # Prepare messages
        messages = []

        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        # Convert stop_sequences to stop for OpenAI compatibility
        if "stop_sequences" in options:
            options["stop"] = options.pop("stop_sequences")

        # Filter out parameters that are not supported by the OpenAI API
        openai_params = {
            k: v
            for k, v in options.items()
            if k
            in [
                "temperature",
                "top_p",
                "n",
                "stream",
                "stop",
                "max_tokens",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "user",
                "response_format",
                "seed",
                "tools",
                "tool_choice",
            ]
        }

        # Send request to OpenAI
        # Type ignore: The OpenAI API expects a specific message format that mypy can't verify
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **openai_params  # type: ignore
        )

        return response.choices[0].message.content or ""

    def _generate_completion(self, prompt: str, **options: Any) -> str:
        """Generate text using the completions API.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the OpenAI API.

        Returns:
            The generated text.
        """
        # Convert stop_sequences to stop for OpenAI compatibility
        if "stop_sequences" in options:
            options["stop"] = options.pop("stop_sequences")

        # Filter out parameters that are not supported by the OpenAI API
        openai_params = {
            k: v
            for k, v in options.items()
            if k
            in [
                "temperature",
                "top_p",
                "n",
                "stream",
                "stop",
                "max_tokens",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "user",
                "suffix",
                "echo",
                "logprobs",
                "best_of",
            ]
        }

        # Send request to OpenAI
        response = self.client.completions.create(
            model=self.model_name, prompt=prompt, **openai_params
        )

        # Ensure we return a string
        result = response.choices[0].text
        return result.strip() if result is not None else ""

    def _is_chat_model(self) -> bool:
        """Determine if the model is a chat model.

        Returns:
            True if the model is a chat model, False otherwise.
        """
        # Most OpenAI models starting with "gpt" are chat models
        return "gpt" in self.model_name.lower()

    def configure(self, **options: Any) -> None:
        """Configure the model with new options.

        Args:
            **options: Configuration options to apply to the model.
                Supported options include:
                - api_key: The OpenAI API key to use.
                - organization: The OpenAI organization ID to use.
                - temperature: Controls randomness in generation.
                - max_tokens: Maximum number of tokens to generate.
                - top_p: Controls diversity via nucleus sampling.
                - frequency_penalty: Reduces repetition of token sequences.
                - presence_penalty: Reduces repetition of topics.

        Raises:
            ModelError: If there is an error configuring the model.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Log configuration attempt
        logger.debug(f"Configuring OpenAI model '{self.model_name}'")

        # Update API key if provided
        if "api_key" in options:
            self.api_key = options["api_key"]
            # Recreate client with new API key
            self.client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
            )
            logger.debug("Updated API key")

        # Update organization if provided
        if "organization" in options:
            self.organization = options["organization"]
            # Recreate client with new organization
            self.client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
            )
            logger.debug("Updated organization")

        # Update other options
        for key, value in options.items():
            if key not in ["api_key", "organization"]:
                self.options[key] = value
                logger.debug(f"Updated option '{key}'")

        logger.debug(f"Successfully configured OpenAI model '{self.model_name}'")

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
        from sifaka.utils.error_handling import model_context, log_error

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
                    "Check if tiktoken is properly installed",
                    "Verify that the model name is supported by tiktoken",
                ],
                metadata={"model_name": self.model_name, "text_length": len(text)},
            ):
                # Get the encoding for the model
                encoding = self._get_encoding()

                # Count tokens
                token_count = len(encoding.encode(text))

                # Log successful token counting
                logger.debug(
                    f"Successfully counted {token_count} tokens for text of length {len(text)}"
                )

                return token_count

        except Exception as e:
            # Log the error
            log_error(e, logger, component="OpenAIModel", operation="token_counting")

            # Raise as ModelError with more context
            raise ModelError(
                message=f"Error counting tokens: {str(e)}",
                component="OpenAIModel",
                operation="token_counting",
                suggestions=[
                    "Check if tiktoken is properly installed",
                    "Verify that the model name is supported by tiktoken",
                    "Try using a different encoding method",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": type(e).__name__,
                    "text_length": len(text),
                },
            )

    def _get_encoding(self) -> Any:
        """Get the encoding for the model.

        Returns:
            The encoding for the model.

        Raises:
            ModelError: If the encoding for the model cannot be determined.
        """
        import logging
        from sifaka.utils.error_handling import model_context, log_error

        logger = logging.getLogger(__name__)

        # Log encoding retrieval attempt
        logger.debug(f"Getting encoding for model '{self.model_name}'")

        try:
            # Try to get the encoding for the specific model
            with model_context(
                model_name=self.model_name,
                operation="get_encoding",
                message_prefix="Failed to get encoding for model",
                suggestions=[
                    "Check if tiktoken is properly installed",
                    "Verify that the model name is supported by tiktoken",
                ],
                metadata={"model_name": self.model_name},
            ):
                try:
                    encoding = tiktoken.encoding_for_model(self.model_name)
                    logger.debug(f"Successfully got encoding for model '{self.model_name}'")
                    return encoding
                except KeyError:
                    # Log fallback attempt
                    logger.debug(
                        f"Encoding not found for model '{self.model_name}', "
                        f"falling back to cl100k_base"
                    )

                    # Fall back to cl100k_base, which is used by gpt-4 and gpt-3.5-turbo
                    encoding = tiktoken.get_encoding("cl100k_base")
                    logger.debug(f"Successfully got fallback encoding 'cl100k_base'")
                    return encoding

        except Exception as e:
            # Log the error
            log_error(e, logger, component="OpenAIModel", operation="get_encoding")

            # Raise as ModelError with more context
            raise ModelError(
                message=f"Error getting encoding: {str(e)}",
                component="OpenAIModel",
                operation="get_encoding",
                suggestions=[
                    "Check if tiktoken is properly installed",
                    "Try using a different encoding method",
                    "Update to the latest version of tiktoken",
                ],
                metadata={"model_name": self.model_name, "error_type": type(e).__name__},
            )


# Register the factory function with the registry
@register_model("openai")
def create_openai_model(model_name: str, **options: Any) -> OpenAIModel:
    """Create an OpenAI model instance.

    This factory function creates an OpenAI model instance with the specified
    model name and options. It is used by the registry system to create models
    without direct imports.

    Args:
        model_name: The name of the OpenAI model to use.
        **options: Additional options to pass to the OpenAI model constructor.

    Returns:
        An OpenAI model instance.

    Raises:
        ConfigurationError: If the OpenAI package is not installed.
        ModelError: If the API key is not provided and not available in the environment.
    """
    import logging
    from sifaka.utils.error_handling import log_error

    logger = logging.getLogger(__name__)

    # Log model creation attempt
    logger.debug(f"Creating OpenAI model with name '{model_name}'")

    try:
        # Create the model
        model = OpenAIModel(model_name=model_name, **options)

        # Log successful model creation
        logger.debug(f"Successfully created OpenAI model with name '{model_name}'")

        return model

    except ConfigurationError as e:
        # Log the error
        log_error(e, logger, component="OpenAIModelFactory", operation="create_model")

        # Re-raise the error with more context
        raise ConfigurationError(
            message=f"Failed to create OpenAI model: {str(e)}",
            component="OpenAIModelFactory",
            operation="create_model",
            suggestions=[
                "Install the OpenAI package with 'pip install openai'",
                "Install the tiktoken package with 'pip install tiktoken'",
            ],
            metadata={"model_name": model_name, "error_type": "ConfigurationError"},
        )

    except ModelError as e:
        # Log the error
        log_error(e, logger, component="OpenAIModelFactory", operation="create_model")

        # Re-raise the error with more context
        raise ModelError(
            message=f"Failed to create OpenAI model: {str(e)}",
            component="OpenAIModelFactory",
            operation="create_model",
            suggestions=[
                "Set the OPENAI_API_KEY environment variable",
                "Pass the API key explicitly as api_key='your-api-key'",
            ],
            metadata={"model_name": model_name, "error_type": "ModelError"},
        )

    except Exception as e:
        # Log the error
        log_error(e, logger, component="OpenAIModelFactory", operation="create_model")

        # Raise as ModelError with more context
        raise ModelError(
            message=f"Unexpected error creating OpenAI model: {str(e)}",
            component="OpenAIModelFactory",
            operation="create_model",
            suggestions=[
                "Check the error message for details",
                "Verify that your options are valid",
            ],
            metadata={"model_name": model_name, "error_type": type(e).__name__},
        )
