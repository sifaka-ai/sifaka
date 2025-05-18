"""
Google Gemini model implementation for Sifaka.

This module provides an implementation of the Model protocol for Google Gemini models.
"""

import os
from typing import Optional, Any, Dict, Union

try:
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPIError, ResourceExhausted, InvalidArgument
    from google.generativeai.types import GenerationConfig as GenAIGenerationConfig

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from sifaka.errors import ModelError, ModelAPIError, ConfigurationError
from sifaka.registry import register_model
from sifaka.interfaces import Model


class GeminiModel(Model):
    """Google Gemini model implementation.

    This class implements the Model protocol for Google Gemini models.

    Attributes:
        model_name: The name of the Gemini model to use.
        api_key: The Google API key to use. If not provided, it will be read from the
            GOOGLE_API_KEY environment variable.
        model: The Gemini model instance.
    """

    # Type annotations for instance variables
    model_name: str
    api_key: Optional[str]
    options: Dict[str, Any]
    model: Any  # genai.GenerativeModel

    def __init__(self, model_name: str, api_key: Optional[str] = None, **options: Any):
        """Initialize the Gemini model.

        Args:
            model_name: The name of the Gemini model to use.
            api_key: The Google API key to use. If not provided, it will be read from the
                GOOGLE_API_KEY environment variable.
            **options: Additional options to pass to the Gemini model.

        Raises:
            ConfigurationError: If the Google Generative AI package is not installed.
            ModelError: If the API key is not provided and not available in the environment.
        """
        import logging
        from sifaka.utils.error_handling import log_error

        logger = logging.getLogger(__name__)

        # Check if Gemini package is available
        if not GEMINI_AVAILABLE:
            error_msg = "Google Generative AI package not installed. Install it with 'pip install google-generativeai'."
            logger.error(error_msg)
            raise ConfigurationError(
                message=error_msg,
                component="GeminiModel",
                operation="initialization",
                suggestions=[
                    "Install the Google Generative AI package with 'pip install google-generativeai'"
                ],
                metadata={"model_name": model_name},
            )

        # Store model configuration
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.options = options

        # Log initialization attempt
        logger.debug(f"Initializing Gemini model '{model_name}'")

        # Check if API key is available
        if not self.api_key:
            error_msg = "Google API key not provided. Either pass it as an argument or set the GOOGLE_API_KEY environment variable."
            logger.error(error_msg)
            raise ModelError(
                message=error_msg,
                component="GeminiModel",
                operation="initialization",
                suggestions=[
                    "Set the GOOGLE_API_KEY environment variable",
                    "Pass the API key explicitly as api_key='your-api-key'",
                ],
                metadata={"model_name": model_name},
            )

        try:
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)

            # Get the model
            self.model = genai.GenerativeModel(model_name=self.model_name)

            # Log successful initialization
            logger.debug(f"Successfully initialized Gemini model '{model_name}'")

        except Exception as e:
            # Log the error
            log_error(e, logger, component="GeminiModel", operation="initialization")

            # Raise as ModelError with more context
            raise ModelError(
                message=f"Error initializing Gemini model: {str(e)}",
                component="GeminiModel",
                operation="initialization",
                suggestions=[
                    "Check if the model name is correct",
                    "Verify that your API key is valid",
                    "Check if the Gemini API is available in your region",
                ],
                metadata={
                    "model_name": model_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options to pass to the Gemini API.
                Supported options include:
                - temperature: Controls randomness. Higher values (e.g., 0.8) make output more random,
                  lower values (e.g., 0.2) make it more deterministic.
                - max_tokens: Maximum number of tokens to generate.
                - top_p: Controls diversity via nucleus sampling.
                - top_k: Controls diversity by limiting to top k tokens.

        Returns:
            The generated text.

        Raises:
            ModelAPIError: If there is an error communicating with the Gemini API.
        """
        import logging
        import time
        from sifaka.utils.error_handling import model_context, log_error

        logger = logging.getLogger(__name__)

        # Merge default options with provided options
        merged_options = {**self.options, **options}

        # Convert max_tokens to max_output_tokens if present
        if "max_tokens" in merged_options:
            merged_options["max_output_tokens"] = merged_options.pop("max_tokens")

        # Log generation attempt
        logger.debug(
            f"Generating text with Gemini model '{self.model_name}', "
            f"prompt length={len(prompt)}, "
            f"temperature={merged_options.get('temperature', 'default')}"
        )

        start_time = time.time()

        try:
            # Use model_context for consistent error handling
            with model_context(
                model_name=self.model_name,
                operation="generation",
                message_prefix="Failed to generate text with Gemini model",
                suggestions=[
                    "Check your API key and ensure it is valid",
                    "Verify that you have sufficient quota",
                    "Check if the model is available in your region",
                ],
                metadata={
                    "model_name": self.model_name,
                    "prompt_length": len(prompt),
                    "temperature": merged_options.get("temperature"),
                    "max_output_tokens": merged_options.get("max_output_tokens"),
                },
            ):
                # Create a proper GenerationConfig object
                generation_config = GenAIGenerationConfig(**merged_options)

                # Generate content with the proper config
                response = self.model.generate_content(prompt, generation_config=generation_config)

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Extract the response text
                response_text: str = response.text

                # Log successful generation
                logger.debug(
                    f"Successfully generated text with Gemini model '{self.model_name}' "
                    f"in {processing_time:.2f}ms, result length={len(response_text)}"
                )

                return response_text

        except ResourceExhausted as e:
            # Log the error
            log_error(e, logger, component="GeminiModel", operation="generation")

            # Raise as ModelAPIError with more context
            raise ModelAPIError(
                message=f"Gemini rate limit exceeded: {str(e)}",
                model_name=self.model_name,
                component="GeminiModel",
                operation="generation",
                suggestions=[
                    "Reduce the frequency of requests",
                    "Implement exponential backoff",
                    "Consider upgrading your Google Cloud plan for higher rate limits",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": "ResourceExhausted",
                    "prompt_length": len(prompt),
                },
            )

        except InvalidArgument as e:
            # Log the error
            log_error(e, logger, component="GeminiModel", operation="generation")

            # Raise as ModelAPIError with more context
            raise ModelAPIError(
                message=f"Invalid argument to Gemini API: {str(e)}",
                model_name=self.model_name,
                component="GeminiModel",
                operation="generation",
                suggestions=[
                    "Check if your prompt contains invalid content",
                    "Verify that your generation parameters are within valid ranges",
                    "Check if the prompt length is within the model's limits",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": "InvalidArgument",
                    "prompt_length": len(prompt),
                },
            )

        except GoogleAPIError as e:
            # Log the error
            log_error(e, logger, component="GeminiModel", operation="generation")

            # Raise as ModelAPIError with more context
            raise ModelAPIError(
                message=f"Gemini API error: {str(e)}",
                model_name=self.model_name,
                component="GeminiModel",
                operation="generation",
                suggestions=[
                    "Check if the model name is correct",
                    "Verify that your API key has access to this model",
                    "Check if there are any issues with the Gemini service",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": "GoogleAPIError",
                    "prompt_length": len(prompt),
                },
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="GeminiModel", operation="generation")

            # Raise as ModelAPIError with more context
            raise ModelAPIError(
                message=f"Unexpected error when calling Gemini API: {str(e)}",
                model_name=self.model_name,
                component="GeminiModel",
                operation="generation",
                suggestions=[
                    "Check the error message for details",
                    "Verify that your request is properly formatted",
                    "Check if there are any issues with the Gemini service",
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
                    "Check if the Gemini API is functioning properly",
                    "Verify that your API key is valid",
                ],
                metadata={"model_name": self.model_name, "text_length": len(text)},
            ):
                # Use Gemini's token counting function through the model
                result = self.model.count_tokens(text)

                # Extract the token count
                token_count: int = result.total_tokens

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Log successful token counting
                logger.debug(
                    f"Successfully counted {token_count} tokens for text of length {len(text)} "
                    f"in {processing_time:.2f}ms"
                )

                return token_count

        except GoogleAPIError as e:
            # Log the error
            log_error(e, logger, component="GeminiModel", operation="token_counting")

            # Raise as ModelError with more context
            raise ModelError(
                message=f"Gemini API error counting tokens: {str(e)}",
                component="GeminiModel",
                operation="token_counting",
                suggestions=[
                    "Check if the model name is correct",
                    "Verify that your API key is valid",
                    "Check if the Gemini API is available in your region",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": "GoogleAPIError",
                    "text_length": len(text),
                },
            )

        except Exception as e:
            # Log the error
            log_error(e, logger, component="GeminiModel", operation="token_counting")

            # Raise as ModelError with more context
            raise ModelError(
                message=f"Error counting tokens: {str(e)}",
                component="GeminiModel",
                operation="token_counting",
                suggestions=[
                    "Check if the Gemini API is functioning properly",
                    "Verify that your API key is valid",
                    "Try with a shorter text",
                ],
                metadata={
                    "model_name": self.model_name,
                    "error_type": type(e).__name__,
                    "text_length": len(text),
                },
            )

    def configure(self, **options: Any) -> None:
        """Configure the model with new options.

        Args:
            **options: Configuration options to apply to the model.
        """
        # Update options
        self.options.update(options)


@register_model("gemini")
def create_gemini_model(model_name: str, **options: Any) -> Model:
    """Create a Google Gemini model instance.

    This factory function creates a Google Gemini model instance with the specified
    model name and options. It is registered with the registry system for
    dependency injection.

    Args:
        model_name: The name of the Gemini model to use.
        **options: Additional options to pass to the Gemini model constructor.

    Returns:
        A Gemini model instance.

    Raises:
        ConfigurationError: If the Google Generative AI package is not installed.
        ModelError: If the API key is not provided and not available in the environment.
    """
    import logging
    from sifaka.utils.error_handling import log_error

    logger = logging.getLogger(__name__)

    # Log model creation attempt
    logger.debug(f"Creating Gemini model with name '{model_name}'")

    try:
        # Create the model
        model = GeminiModel(model_name=model_name, **options)

        # Log successful model creation
        logger.debug(f"Successfully created Gemini model with name '{model_name}'")

        return model

    except ConfigurationError as e:
        # Log the error
        log_error(e, logger, component="GeminiModelFactory", operation="create_model")

        # Re-raise the error with more context
        raise ConfigurationError(
            message=f"Failed to create Gemini model: {str(e)}",
            component="GeminiModelFactory",
            operation="create_model",
            suggestions=[
                "Install the Google Generative AI package with 'pip install google-generativeai'"
            ],
            metadata={"model_name": model_name, "error_type": "ConfigurationError"},
        )

    except ModelError as e:
        # Log the error
        log_error(e, logger, component="GeminiModelFactory", operation="create_model")

        # Re-raise the error with more context
        raise ModelError(
            message=f"Failed to create Gemini model: {str(e)}",
            component="GeminiModelFactory",
            operation="create_model",
            suggestions=[
                "Set the GOOGLE_API_KEY environment variable",
                "Pass the API key explicitly as api_key='your-api-key'",
            ],
            metadata={"model_name": model_name, "error_type": "ModelError"},
        )

    except Exception as e:
        # Log the error
        log_error(e, logger, component="GeminiModelFactory", operation="create_model")

        # Raise as ModelError with more context
        raise ModelError(
            message=f"Unexpected error creating Gemini model: {str(e)}",
            component="GeminiModelFactory",
            operation="create_model",
            suggestions=[
                "Check the error message for details",
                "Verify that your options are valid",
            ],
            metadata={"model_name": model_name, "error_type": type(e).__name__},
        )
