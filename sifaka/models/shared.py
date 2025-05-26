"""Shared utilities and base classes for model implementations.

This module provides common functionality that can be shared across different
model provider implementations to reduce code duplication.
"""

from typing import Any, Callable, Dict, List, Optional

from sifaka.core.thought import Thought
from sifaka.utils.error_handling import ConfigurationError, ModelError, log_error, model_context
from sifaka.utils.logging import get_logger
from sifaka.utils.mixins import APIKeyMixin, ContextAwareMixin

logger = get_logger(__name__)


class BaseModelImplementation(ContextAwareMixin, APIKeyMixin):
    """Base class for model implementations with common functionality.

    This class provides shared functionality for model implementations including:
    - API key management
    - Error handling patterns
    - Logging patterns
    - Context awareness
    - Common configuration validation

    Subclasses should implement the abstract methods and can override
    the default implementations as needed.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        provider_name: str = "Unknown",
        env_var_name: str = "API_KEY",
        required_packages: Optional[List[str]] = None,
        api_key_required: bool = True,
        **options: Any,
    ):
        """Initialize the base model implementation.

        Args:
            model_name: Name of the model.
            api_key: API key for the provider.
            provider_name: Name of the provider for error messages.
            env_var_name: Environment variable name for API key.
            required_packages: List of required package names.
            api_key_required: Whether an API key is required.
            **options: Additional model options.
        """
        super().__init__()

        self.model_name = model_name
        self.provider_name = provider_name
        self.options = options

        # Check required packages
        if required_packages:
            self._check_required_packages(required_packages)

        # Get API key using the mixin
        self.api_key = self.get_api_key(
            api_key=api_key,
            env_var_name=env_var_name,
            provider_name=provider_name,
            required=api_key_required,
        )

        logger.debug(f"Initialized {provider_name} model '{model_name}'")

    def _check_required_packages(self, required_packages: List[str]) -> None:
        """Check if required packages are available.

        Args:
            required_packages: List of package names to check.

        Raises:
            ConfigurationError: If required packages are not available.
        """
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            package_list = "', '".join(missing_packages)
            raise ConfigurationError(
                f"Required packages not available: '{package_list}'",
                component=self.provider_name,
                operation="initialization",
                suggestions=[
                    f"Install missing packages: pip install {' '.join(missing_packages)}",
                    f"Check the {self.provider_name} documentation for installation instructions",
                ],
            )

    def _handle_api_error(self, error: Exception, operation: str = "generation") -> None:
        """Handle API errors with consistent logging and error transformation.

        Args:
            error: The original exception.
            operation: The operation that failed.

        Raises:
            ModelError: Transformed error with additional context.
        """
        error_message = str(error)

        # Log the error with context
        log_error(error, component=self.provider_name, operation=operation, include_traceback=True)

        # Create suggestions based on common error patterns
        suggestions = []

        if "api key" in error_message.lower() or "unauthorized" in error_message.lower():
            suggestions.extend(
                [
                    f"Check your {self.provider_name} API key",
                    "Ensure the API key has the necessary permissions",
                    "Verify the API key is not expired",
                ]
            )
        elif "rate limit" in error_message.lower() or "quota" in error_message.lower():
            suggestions.extend(
                [
                    "Wait before retrying the request",
                    "Check your API usage limits",
                    "Consider upgrading your API plan",
                ]
            )
        elif "connection" in error_message.lower() or "timeout" in error_message.lower():
            suggestions.extend(
                [
                    "Check your internet connection",
                    "Retry the request after a short delay",
                    "Verify the API endpoint is accessible",
                ]
            )
        else:
            suggestions.extend(
                [
                    f"Check the {self.provider_name} API documentation",
                    "Verify your request parameters are correct",
                    "Check the service status page",
                ]
            )

        # Raise a ModelError with enhanced context
        raise ModelError(
            f"{self.provider_name} API error during {operation}: {error_message}",
            component=self.provider_name,
            operation=operation,
            suggestions=suggestions,
            metadata={"model_name": self.model_name, "original_error": type(error).__name__},
        ) from error

    def _validate_generate_params(self, prompt: str, **options: Any) -> Dict[str, Any]:
        """Validate and normalize generation parameters.

        Args:
            prompt: The input prompt.
            **options: Generation options.

        Returns:
            Validated and normalized options.

        Raises:
            ModelError: If parameters are invalid.
        """
        if not prompt or not prompt.strip():
            raise ModelError(
                "Empty or invalid prompt provided",
                component=self.provider_name,
                operation="parameter_validation",
                suggestions=[
                    "Provide a non-empty prompt string",
                    "Check that the prompt contains meaningful content",
                ],
            )

        # Merge with default options
        validated_options = {**self.options, **options}

        # Validate common parameters
        if "temperature" in validated_options:
            temp = validated_options["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                raise ModelError(
                    f"Invalid temperature value: {temp}. Must be between 0 and 2.",
                    component=self.provider_name,
                    operation="parameter_validation",
                    suggestions=["Use a temperature value between 0.0 and 2.0"],
                )

        if "max_tokens" in validated_options:
            max_tokens = validated_options["max_tokens"]
            if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
                raise ModelError(
                    f"Invalid max_tokens value: {max_tokens}. Must be a positive integer.",
                    component=self.provider_name,
                    operation="parameter_validation",
                    suggestions=["Use a positive integer for max_tokens"],
                )

        return validated_options

    def generate(self, prompt: str, **options: Any) -> str:
        """Generate text from a prompt with validation and error handling.

        This method provides the standard interface for text generation. It validates
        parameters, handles errors consistently, and delegates to the subclass-specific
        _generate_impl method.

        Args:
            prompt: The prompt to generate text from.
            **options: Additional options for generation.

        Returns:
            The generated text.

        Raises:
            ModelError: If parameters are invalid or generation fails.
        """
        # Validate and normalize parameters
        validated_options = self._validate_generate_params(prompt, **options)

        # Use model context for consistent error handling
        with model_context(
            model_name=self.model_name,
            operation="generation",
            message_prefix=f"Failed to generate text with {self.provider_name} model",
            metadata={
                "model_name": self.model_name,
                "prompt_length": len(prompt),
                "temperature": validated_options.get("temperature"),
                "max_tokens": validated_options.get("max_tokens"),
            },
        ):
            # Delegate to subclass implementation
            return self._generate_impl(prompt, **validated_options)

    def _generate_impl(self, prompt: str, **options: Any) -> str:
        """Generate text implementation to be overridden by subclasses.

        This method should be implemented by subclasses to provide the actual
        text generation logic. It will be called by the base generate method
        after parameter validation and error context setup.

        Args:
            prompt: The validated prompt to generate text from.
            **options: Validated and normalized generation options.

        Returns:
            The generated text.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _generate_impl method")

    def generate_with_thought(self, thought: Thought, **options: Any) -> tuple[str, str]:
        """Generate text using a Thought container with context awareness.

        This method leverages the ContextAwareMixin to prepare context from
        the Thought container and includes it in the generation process.

        Args:
            thought: The Thought container with context for generation.
            **options: Additional options for generation.

        Returns:
            A tuple of (generated_text, actual_prompt_used).
        """
        with model_context(
            model_name=self.model_name,
            operation="thought_generation",
            message_prefix=f"Failed to generate text with {self.provider_name}",
        ):
            # Prepare contextualized prompt using the mixin
            contextualized_prompt = self._build_contextualized_prompt(thought)

            # Generate text with the contextualized prompt
            generated_text = self.generate(contextualized_prompt, **options)

            return generated_text, contextualized_prompt


def create_factory_function(
    model_class: type,
    provider_name: str,
    env_var_name: str,
    required_packages: Optional[List[str]] = None,
) -> Callable[[str], Any]:
    """Create a standardized factory function for a model class.

    This function generates a factory function that follows the same pattern
    across all model providers, reducing code duplication.

    Args:
        model_class: The model class to instantiate.
        provider_name: Name of the provider for error messages.
        env_var_name: Environment variable name for API key.
        required_packages: List of required package names.

    Returns:
        A factory function that creates model instances.
    """

    def factory_function(model_name: str, **kwargs: Any) -> Any:
        """Factory function for creating model instances.

        Args:
            model_name: Name of the model to create.
            **kwargs: Additional arguments for the model constructor.

        Returns:
            A model instance.

        Raises:
            ConfigurationError: If required packages are not available.
            ModelError: If there's an error creating the model.
        """
        logger.debug(f"Creating {provider_name} model with name '{model_name}'")

        try:
            model = model_class(
                model_name=model_name,
                provider_name=provider_name,
                env_var_name=env_var_name,
                required_packages=required_packages,
                **kwargs,
            )

            logger.debug(f"Successfully created {provider_name} model: {model_name}")
            return model

        except Exception as e:
            logger.error(f"Failed to create {provider_name} model '{model_name}': {e}")
            raise

    # Set function metadata for better debugging
    factory_function.__name__ = f"create_{provider_name.lower()}_model"
    factory_function.__doc__ = f"""Create a {provider_name} model instance.

    Args:
        model_name: Name of the {provider_name} model.
        **kwargs: Additional arguments for the model constructor.

    Returns:
        A {provider_name} model instance.
    """

    return factory_function
