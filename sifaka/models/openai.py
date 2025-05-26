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
import time
from typing import Any, Optional

try:
    import tiktoken
    from openai import APIConnectionError, APIError, OpenAI, RateLimitError

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from sifaka.core.thought import Thought
from sifaka.utils.error_handling import (
    ConfigurationError,
    ModelAPIError,
    ModelError,
    log_error,
    model_context,
)
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import APIKeyMixin, ContextAwareMixin

# Configure logger
logger = get_logger(__name__)


class OpenAIModel(APIKeyMixin, ContextAwareMixin):
    """OpenAI model implementation for generating text and counting tokens.

    This class implements the Model protocol for OpenAI models, supporting both
    the chat completions API and the completions API. It automatically selects
    the appropriate API based on the model name.

    For chat models (those with "gpt" in the name), it uses the chat completions API
    and supports system messages. For other models, it uses the completions API.

    The class also provides methods for counting tokens and configuring the model
    after initialization.

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
            model_name: The name of the OpenAI model to use (e.g., "gpt-4", "text-davinci-003").
            api_key: Optional API key to use. If not provided, it will be read from the
                OPENAI_API_KEY environment variable.
            organization: Optional organization ID to use. If not provided, it will be read
                from the OPENAI_ORGANIZATION environment variable.
            **options: Additional options to pass to the OpenAI API, such as:
                - temperature: Controls randomness in generation (0.0 to 1.0).
                - max_tokens: Maximum number of tokens to generate.
                - top_p: Controls diversity via nucleus sampling.
                - presence_penalty: Penalizes repeated tokens.
                - frequency_penalty: Penalizes frequent tokens.

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

        # Initialize mixins
        super().__init__()

        # Store model name and options
        self.model_name = model_name
        self.options = options

        # Get API key using the mixin (eliminates duplicate code)
        self.api_key = self.get_api_key(
            api_key=api_key, env_var_name="OPENAI_API_KEY", provider_name="OpenAI", required=True
        )

        # Get organization from parameter or environment variable
        self.organization = organization or os.environ.get("OPENAI_ORGANIZATION")

        # Initialize the OpenAI client (sync)
        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.organization,
        )

        # Initialize the async OpenAI client
        from openai import AsyncOpenAI

        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization,
        )

        logger.debug(f"Initialized OpenAI model '{model_name}'")

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt using the OpenAI API.

        This method automatically selects the appropriate API (chat completions or completions)
        based on the model name, unless overridden with the use_completion_api option.

        For chat models (those with "gpt" in the name), it uses the chat completions API
        and supports system messages. For other models, it uses the completions API.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the OpenAI API, such as:
                - temperature: Controls randomness in generation (0.0 to 1.0).
                - max_tokens: Maximum number of tokens to generate.
                - system_message: System message for chat models.
                - use_completion_api: Force use of the completion API instead of chat.
                - stop_sequences: Sequences where the API will stop generating further tokens.

        Returns:
            The generated text.

        Raises:
            ModelAPIError: If there is an error calling the OpenAI API.
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

        # Determine if we should use the completion API
        use_completion_api = merged_options.pop("use_completion_api", False)

        # Extract system message if provided
        system_message = merged_options.pop("system_message", None)

        # Log generation attempt
        logger.debug(
            f"Generating text with OpenAI model '{self.model_name}', "
            f"prompt length={len(prompt)}, "
            f"temperature={merged_options.get('temperature', 'default')}"
        )

        start_time = time.time()

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

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Log successful generation
                logger.debug(
                    f"Successfully generated text with OpenAI model '{self.model_name}' "
                    f"in {processing_time:.2f}ms, result length={len(result)}"
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

            # Raise as ModelError with more context
            raise ModelError(
                message=f"Error generating text with OpenAI model: {str(e)}",
                component="OpenAIModel",
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
            A tuple of (generated_text, actual_prompt_used).

        Raises:
            ModelAPIError: If there is an error calling the OpenAI API.
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
            logger.debug(f"OpenAIModel using context: {context_summary}")

        # Generate text using the contextualized prompt and options
        generated_text = self.generate(full_prompt, **generation_options)
        return generated_text, full_prompt

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the OpenAI tokenizer.

        This method uses the tiktoken library to count tokens according to the
        tokenization scheme of the specified model. Different models may tokenize
        text differently, so token counts may vary between models.

        Args:
            text: The text to count tokens in.

        Returns:
            The number of tokens in the text according to the model's tokenization scheme.

        Raises:
            ModelError: If there is an error counting tokens, such as an unsupported model
                or issues with the tiktoken library.

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
            The tiktoken encoding object for the model.

        Raises:
            ModelError: If the encoding for the model cannot be determined,
                such as if tiktoken is not installed or the model is not supported.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Log encoding retrieval attempt
        logger.debug(f"Getting encoding for model '{self.model_name}'")

        try:
            # Try to get encoding for the specific model
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
                logger.debug(f"Using model-specific encoding for '{self.model_name}'")
                return encoding
            except KeyError:
                # If model-specific encoding is not available, use a default encoding
                logger.debug(
                    f"Model-specific encoding not available for '{self.model_name}', "
                    "using default encoding cl100k_base"
                )
                return tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            # Log the error
            log_error(e, logger, component="OpenAIModel", operation="get_encoding")

            # Raise as ModelError with more context
            raise ModelError(
                message=f"Error getting encoding for model '{self.model_name}': {str(e)}",
                component="OpenAIModel",
                operation="get_encoding",
                suggestions=[
                    "Check if tiktoken is properly installed",
                    "Verify that the model name is supported by tiktoken",
                    "Try using a different encoding method",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": type(e).__name__,
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
            prompt: The prompt to generate text from.
            system_message: Optional system message to include at the
                beginning of the conversation.
            **options: Additional options to pass to the OpenAI API.

        Returns:
            The generated text from the assistant's response.
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
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **openai_params,  # type: ignore
        )

        # Ensure we return a string
        result = response.choices[0].message.content
        return str(result.strip() if result is not None else "")

    def _generate_completion(self, prompt: str, **options: Any) -> str:
        """Generate text using the OpenAI completions API.

        This internal method is used by the generate method when the model is not a chat model
        (those without "gpt" in the name) or when use_completion_api is set to True.

        It sends the prompt directly to the OpenAI completions API without formatting
        it into the chat message format.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the OpenAI API.

        Returns:
            The generated text from the completion response.
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
        return str(result.strip() if result is not None else "")

    def _is_chat_model(self) -> bool:
        """Determine if the model is a chat model based on its name.

        This internal method is used by the generate method to determine whether
        to use the chat completions API or the completions API.

        Currently, it considers any model with "gpt" in its name to be a chat model.

        Returns:
            True if the model is a chat model, False otherwise.
        """
        # Most OpenAI models starting with "gpt" are chat models
        return "gpt" in self.model_name.lower()

    # Internal async methods (implementing the Model protocol)
    async def _generate_async(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt using the OpenAI API asynchronously.

        This is the internal async implementation that provides the same functionality
        as the sync generate method but with non-blocking I/O.
        """
        # Merge default options with provided options
        merged_options = {**self.options, **options}

        # Determine if we should use the completion API
        use_completion_api = merged_options.pop("use_completion_api", False)

        # Extract system message if provided
        system_message = merged_options.pop("system_message", None)

        # Log generation attempt
        logger.debug(
            f"Generating text with OpenAI model '{self.model_name}' (async), "
            f"prompt length={len(prompt)}, "
            f"temperature={merged_options.get('temperature', 'default')}"
        )

        start_time = time.time()

        try:
            # Use the appropriate API based on the model and options
            with model_context(
                model_name=self.model_name,
                operation="generation_async",
                message_prefix="Failed to generate text with OpenAI model (async)",
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
                    result = await self._generate_chat_async(
                        prompt, system_message, **merged_options
                    )
                else:
                    result = await self._generate_completion_async(prompt, **merged_options)

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Log successful generation
                logger.debug(
                    f"Successfully generated text with OpenAI model '{self.model_name}' (async) "
                    f"in {processing_time:.2f}ms, result length={len(result)}"
                )

                return result

        except Exception as e:
            # Handle errors the same way as sync version
            # (Error handling code would be similar to sync version)
            logger.error(f"Async generation error: {e}")
            raise

    async def _generate_with_thought_async(
        self, thought: "Thought", **options: Any
    ) -> tuple[str, str]:
        """Generate text using a Thought container asynchronously."""
        # Use mixin to build contextualized prompt
        full_prompt = self._build_contextualized_prompt(thought, max_docs=5)

        # Add system_prompt to options if available
        generation_options = options.copy()
        if thought.system_prompt:
            generation_options["system_message"] = thought.system_prompt

        # Log context usage
        if self._has_context(thought):
            context_summary = self._get_context_summary(thought)
            logger.debug(f"OpenAIModel using context (async): {context_summary}")

        # Generate text using the contextualized prompt and options
        generated_text = await self._generate_async(full_prompt, **generation_options)
        return generated_text, full_prompt

    async def _count_tokens_async(self, text: str) -> int:
        """Count tokens in text asynchronously."""
        # Token counting is CPU-bound, so we can just call the sync version
        # In a real implementation, you might want to run this in a thread pool
        return self.count_tokens(text)

    async def _generate_chat_async(
        self, prompt: str, system_message: Optional[str] = None, **options: Any
    ) -> str:
        """Generate text using the OpenAI chat completions API asynchronously."""
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

        # Send request to OpenAI asynchronously
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **openai_params,  # type: ignore
        )

        # Ensure we return a string
        result = response.choices[0].message.content
        return str(result.strip() if result is not None else "")

    async def _generate_completion_async(self, prompt: str, **options: Any) -> str:
        """Generate text using the OpenAI completions API asynchronously."""
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

        # Send request to OpenAI asynchronously
        response = await self.async_client.completions.create(
            model=self.model_name, prompt=prompt, **openai_params
        )

        # Ensure we return a string
        result = response.choices[0].text
        return str(result.strip() if result is not None else "")

    def configure(self, **options: Any) -> None:
        """Update the model configuration.

        This method allows updating the model configuration after initialization,
        including the API key, organization, and other options.

        Args:
            **options: New configuration options to set, such as:
                - api_key: New API key to use.
                - organization: New organization ID to use.
                - temperature: New temperature value.
                - max_tokens: New maximum tokens value.

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
            # Recreate both sync and async clients with new API key
            from openai import AsyncOpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
            )
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
            )
            logger.debug("Updated API key")

        # Update organization if provided
        if "organization" in options:
            self.organization = options["organization"]
            # Recreate both sync and async clients with new organization
            from openai import AsyncOpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
            )
            self.async_client = AsyncOpenAI(
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


def create_openai_model(
    model_name: str,
    **options: Any,
) -> OpenAIModel:
    """Create an OpenAI model instance.

    This factory function creates an OpenAI model instance with the specified
    model name and options. It handles error logging and provides a consistent
    interface for creating OpenAI models.

    Args:
        model_name: The name of the OpenAI model to use (e.g., "gpt-4", "text-davinci-003").
        **options: Additional options to pass to the OpenAI model constructor, such as:
            - api_key: The OpenAI API key to use.
            - organization: The OpenAI organization ID to use.
            - temperature: Controls randomness in generation (0.0 to 1.0).
            - max_tokens: Maximum number of tokens to generate.

    Returns:
        An OpenAI model instance implementing the Model protocol.

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
    except Exception as e:
        # Log the error
        log_error(e, logger, component="OpenAIModel", operation="creation")

        # Re-raise the exception
        raise
