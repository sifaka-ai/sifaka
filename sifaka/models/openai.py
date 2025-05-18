"""OpenAI model implementation for Sifaka.

This module provides an implementation of the Model protocol for OpenAI models,
supporting both the chat completions API and the completions API. It handles
token counting, error handling, and configuration for OpenAI models.

The OpenAIModel class implements the Model protocol and provides methods for
generating text and counting tokens. It also provides a configure method for
updating model options after initialization.

Example:
    ```python
    from sifaka.models.openai import OpenAIModel, create_openai_model

    # Create a model directly
    model1 = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

    # Create a model using the factory function
    model2 = create_openai_model(model_name="gpt-4", api_key="your-api-key")

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
from typing import Any, Optional

try:
    import tiktoken
    from openai import APIConnectionError, APIError, OpenAI, RateLimitError

    # No need to import specific types for now

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from sifaka.errors import ConfigurationError, ModelAPIError, ModelError
from sifaka.registry import register_model


class OpenAIModel:
    """OpenAI model implementation for generating text and counting tokens.

    This class implements the Model protocol for OpenAI models, supporting both
    the chat completions API and the completions API. It automatically selects
    the appropriate API based on the model name.

    For chat models (those with "gpt" in the name), it uses the chat completions API
    and supports system messages. For other models, it uses the completions API.

    Attributes:
        model_name (str): The name of the OpenAI model to use (e.g., "gpt-4", "text-davinci-003").
        api_key (str): The OpenAI API key to use. If not provided during initialization,
            it will be read from the OPENAI_API_KEY environment variable.
        organization (Optional[str]): The OpenAI organization ID to use.
        client (OpenAI): The OpenAI client instance used to make API calls.
        options (Dict[str, Any]): Additional options to use when generating text.

    Example:
        ```python
        from sifaka.models.openai import OpenAIModel

        # Create a model
        model = OpenAIModel(
            model_name="gpt-4",
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
        organization: Optional[str] = None,
        **options: Any,
    ):
        """Initialize the OpenAI model with the specified parameters.

        Args:
            model_name (str): The name of the OpenAI model to use (e.g., "gpt-4", "text-davinci-003").
            api_key (Optional[str]): The OpenAI API key to use. If not provided, it will be read
                from the OPENAI_API_KEY environment variable.
            organization (Optional[str]): The OpenAI organization ID to use. This is only
                needed for users who belong to multiple organizations.
            **options (Any): Additional options to use when generating text, such as:
                - temperature: Controls randomness (0.0 to 1.0)
                - max_tokens: Maximum number of tokens to generate
                - top_p: Controls diversity via nucleus sampling
                - frequency_penalty: Reduces repetition of token sequences
                - presence_penalty: Reduces repetition of topics

        Raises:
            ConfigurationError: If the OpenAI package is not installed.
            ModelError: If the API key is not provided and not available in the environment.

        Example:
            ```python
            # Create a model with API key from environment variable
            model1 = OpenAIModel(model_name="gpt-4")

            # Create a model with explicit API key and options
            model2 = OpenAIModel(
                model_name="gpt-4",
                api_key="your-api-key",
                organization="your-org-id",
                temperature=0.7,
                max_tokens=500
            )
            ```
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
        """Generate text from a prompt using the OpenAI API.

        This method automatically selects the appropriate API (chat completions or completions)
        based on the model name, unless overridden with the use_completion_api option.

        For chat models (those with "gpt" in the name), it uses the chat completions API
        and supports system messages. For other models, it uses the completions API.

        Args:
            prompt (str): The prompt to generate text from.
            **options (Any): Additional options to pass to the OpenAI API, including:
                - temperature (float): Controls randomness. Higher values (e.g., 0.8) make output
                  more random, lower values (e.g., 0.2) make it more deterministic. Default is 1.0.
                - max_tokens (int): Maximum number of tokens to generate.
                - top_p (float): Controls diversity via nucleus sampling. Default is 1.0.
                - frequency_penalty (float): Reduces repetition of token sequences. Range is -2.0 to 2.0.
                - presence_penalty (float): Reduces repetition of topics. Range is -2.0 to 2.0.
                - stop (List[str]): Sequences where the API will stop generating further tokens.
                - system_message (str): A system message to include at the beginning of the conversation
                  (only for chat models).
                - use_completion_api (bool): Force using the completion API instead of chat API.
                - stop_sequences (List[str]): Alternative name for stop parameter.

        Returns:
            str: The generated text.

        Raises:
            ModelAPIError: If there is an error communicating with the OpenAI API,
                such as rate limits, connection issues, or invalid parameters.

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
        import logging

        from sifaka.utils.error_handling import log_error, model_context

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
        """Generate text using the OpenAI chat completions API.

        This internal method is used by the generate method when the model is a chat model
        (those with "gpt" in the name) or when use_completion_api is set to False.

        It formats the prompt and system message into the chat message format required
        by the OpenAI chat completions API.

        Args:
            prompt (str): The prompt to generate text from.
            system_message (Optional[str]): Optional system message to include at the
                beginning of the conversation.
            **options (Any): Additional options to pass to the OpenAI API.

        Returns:
            str: The generated text from the assistant's response.
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
        """Generate text using the OpenAI completions API.

        This internal method is used by the generate method when the model is not a chat model
        (those without "gpt" in the name) or when use_completion_api is set to True.

        It sends the prompt directly to the OpenAI completions API without formatting
        it into the chat message format.

        Args:
            prompt (str): The prompt to generate text from.
            **options (Any): Additional options to pass to the OpenAI API.

        Returns:
            str: The generated text from the completion response.
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
        """Determine if the model is a chat model based on its name.

        This internal method is used by the generate method to determine whether
        to use the chat completions API or the completions API.

        Currently, it considers any model with "gpt" in its name to be a chat model.

        Returns:
            bool: True if the model is a chat model, False otherwise.
        """
        # Most OpenAI models starting with "gpt" are chat models
        return "gpt" in self.model_name.lower()

    def configure(self, **options: Any) -> None:
        """Configure the model with new options after initialization.

        This method allows updating the model's configuration after it has been
        initialized. It can be used to change the API key, organization, or any
        of the generation options.

        If the API key or organization is updated, the OpenAI client will be
        recreated with the new values.

        Args:
            **options (Any): Configuration options to apply to the model, including:
                - api_key (str): The OpenAI API key to use.
                - organization (str): The OpenAI organization ID to use.
                - temperature (float): Controls randomness in generation (0.0 to 1.0).
                - max_tokens (int): Maximum number of tokens to generate.
                - top_p (float): Controls diversity via nucleus sampling (0.0 to 1.0).
                - frequency_penalty (float): Reduces repetition of token sequences (-2.0 to 2.0).
                - presence_penalty (float): Reduces repetition of topics (-2.0 to 2.0).
                - stop (List[str]): Sequences where the API will stop generating further tokens.
                - system_message (str): A system message to include at the beginning of the conversation.

        Raises:
            ModelError: If there is an error configuring the model, such as an invalid API key.

        Example:
            ```python
            # Create a model
            model = OpenAIModel(model_name="gpt-4", api_key="your-api-key")

            # Update configuration
            model.configure(
                temperature=0.5,
                max_tokens=1000,
                presence_penalty=0.2
            )

            # Update API key
            model.configure(api_key="your-new-api-key")
            ```
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
        """Count tokens in text using the OpenAI tokenizer.

        This method uses the tiktoken library to count tokens according to the
        tokenization scheme of the specified model. Different models may tokenize
        text differently, so token counts may vary between models.

        Args:
            text (str): The text to count tokens in.

        Returns:
            int: The number of tokens in the text according to the model's tokenization scheme.

        Raises:
            ModelError: If there is an error counting tokens, such as an unsupported model
                or issues with the tiktoken library.

        Example:
            ```python
            # Count tokens in a string
            token_count = model.count_tokens("This is a test.")
            print(f"Token count: {token_count}")

            # Count tokens in a longer text
            with open("document.txt", "r") as f:
                content = f.read()
                token_count = model.count_tokens(content)
                print(f"Document token count: {token_count}")
            ```
        """
        import logging

        from sifaka.utils.error_handling import log_error, model_context

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
        """Get the tiktoken encoding for the model.

        This internal method is used by the count_tokens method to get the
        appropriate tokenizer for the model. It tries to get the encoding
        specifically for the model, and falls back to a default encoding
        if the model-specific encoding is not available.

        Returns:
            Any: The tiktoken encoding object for the model.

        Raises:
            ModelError: If the encoding for the model cannot be determined,
                such as if tiktoken is not installed or the model is not supported.
        """
        import logging

        from sifaka.utils.error_handling import log_error, model_context

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
                    logger.debug("Successfully got fallback encoding 'cl100k_base'")
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
                metadata={
                    "model_name": self.model_name,
                    "error_type": type(e).__name__,
                },
            )


# Register the factory function with the registry
@register_model("openai")
def create_openai_model(model_name: str, **options: Any) -> OpenAIModel:
    """Create an OpenAI model instance.

    This factory function creates an OpenAI model instance with the specified
    model name and options. It is registered with the registry system under
    the "openai" prefix, allowing it to be used with the create_model function.

    It provides consistent error handling and logging for model creation.

    Args:
        model_name (str): The name of the OpenAI model to use (e.g., "gpt-4", "text-davinci-003").
        **options (Any): Additional options to pass to the OpenAI model constructor, such as:
            - api_key (str): The OpenAI API key to use.
            - organization (str): The OpenAI organization ID to use.
            - temperature (float): Controls randomness in generation (0.0 to 1.0).
            - max_tokens (int): Maximum number of tokens to generate.

    Returns:
        OpenAIModel: An OpenAI model instance implementing the Model protocol.

    Raises:
        ConfigurationError: If the OpenAI package is not installed.
        ModelError: If the API key is not provided and not available in the environment,
            or if there is another error creating the model.

    Example:
        ```python
        from sifaka.models.openai import create_openai_model

        # Create a model using the factory function
        model = create_openai_model(
            model_name="gpt-4",
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
