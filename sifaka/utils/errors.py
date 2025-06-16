"""Comprehensive error handling for Sifaka.

This module provides a hierarchy of exception types for different error conditions
in the Sifaka workflow, along with utilities for error context and suggestions.

Exception Hierarchy:
    SifakaError (base)
    ├── ValidationError (validation failures)
    ├── CritiqueError (critic execution failures)
    ├── GraphExecutionError (graph workflow failures)
    └── ConfigurationError (configuration issues)

Each exception type includes:
- Detailed error messages
- Context information
- Actionable suggestions for resolution
- Error codes for programmatic handling
"""

import os
import re
from typing import Any, Dict, List, Optional


class ErrorMessageBuilder:
    """Helper class for building user-friendly error messages with context-aware suggestions."""

    @staticmethod
    def format_api_key_error(provider: str, key_name: str) -> tuple[str, List[str]]:
        """Format API key related error messages.

        Args:
            provider: Name of the provider (e.g., "OpenAI", "Anthropic")
            key_name: Environment variable name (e.g., "OPENAI_API_KEY")

        Returns:
            Tuple of (error_message, suggestions)
        """
        message = f"🔑 {provider} API key is missing or invalid"

        # Check if key exists but might be invalid
        current_key = os.getenv(key_name, "")
        if current_key:
            if len(current_key) < 10:
                key_status = "appears too short"
            elif not current_key.startswith(("sk-", "claude-", "gsk_")):
                key_status = "doesn't match expected format"
            else:
                key_status = "exists but may be invalid"
            message += f" (key {key_status})"

        suggestions = [
            f"💡 Quick fix: export {key_name}='your-actual-key-here'",
            f"🌐 Get a new API key from {provider.lower()}.com/api-keys",
            (
                f"🔍 Current key status: {current_key[:8]}..."
                if current_key
                else f"❌ No {key_name} found in environment"
            ),
            "✅ Test your key with a simple API call first",
            f"🔄 Alternative: Try sifaka.presets.quick_draft() with a different model",
        ]
        return message, suggestions

    @staticmethod
    def format_model_error(model_name: str, error_type: str) -> tuple[str, List[str]]:
        """Format model-related error messages.

        Args:
            model_name: Name of the model that failed
            error_type: Type of error (e.g., "connection", "rate_limit", "invalid_model")

        Returns:
            Tuple of (error_message, suggestions)
        """
        if error_type == "connection":
            message = f"🌐 Cannot connect to model '{model_name}'"
            suggestions = [
                "🔍 Check your internet connection",
                "✏️ Verify model name format: 'provider:model' (e.g., 'openai:gpt-4')",
                "⏰ Try again in a few moments (temporary network issue)",
                "🔄 Quick alternative: await sifaka.presets.quick_draft('your prompt')",
                "📋 Working models: openai:gpt-4, anthropic:claude-3-sonnet, openai:gpt-4o-mini",
            ]
        elif error_type == "rate_limit":
            message = f"⏱️ Rate limit exceeded for model '{model_name}'"
            suggestions = [
                "⏳ Wait 1-2 minutes before trying again",
                "💰 Consider upgrading your API plan for higher limits",
                "🔄 Try a faster model: await sifaka.presets.quick_draft() uses gpt-4o-mini",
                "🛠️ Use exponential backoff: retry after 1s, 2s, 4s, 8s...",
                "📊 Check your usage at the provider's dashboard",
            ]
        elif error_type == "invalid_model":
            message = f"❌ Model '{model_name}' is not available"
            suggestions = [
                "✏️ Check spelling - use format 'provider:model'",
                "🌍 Verify model availability in your region",
                "✅ Try these working models:",
                "   • openai:gpt-4 (high quality)",
                "   • openai:gpt-4o-mini (fast & cheap)",
                "   • anthropic:claude-3-sonnet (balanced)",
                "📚 Check provider docs for complete model list",
            ]
        else:
            message = f"⚠️ Model error '{model_name}': {error_type}"
            suggestions = [
                "🔧 Check model configuration and API credentials",
                "🔄 Try a different model as fallback",
                "🌐 Check provider status page for outages",
                "💡 Quick test: await sifaka.presets.quick_draft('test prompt')",
            ]

        return message, suggestions

    @staticmethod
    def format_validation_error(
        validator_name: str, issue: str, current_value: Any, expected: str
    ) -> tuple[str, List[str]]:
        """Format validation error messages with specific guidance.

        Args:
            validator_name: Name of the validator that failed
            issue: Description of the validation issue
            current_value: Current value that failed validation
            expected: Description of what was expected

        Returns:
            Tuple of (error_message, suggestions)
        """
        message = f"{validator_name} validation failed: {issue}"

        # Context-aware suggestions based on validator type
        if "length" in validator_name.lower():
            suggestions = ErrorMessageBuilder._get_length_suggestions(
                issue, current_value, expected
            )
        elif "content" in validator_name.lower():
            suggestions = ErrorMessageBuilder._get_content_suggestions(
                issue, current_value, expected
            )
        elif "format" in validator_name.lower():
            suggestions = ErrorMessageBuilder._get_format_suggestions(
                issue, current_value, expected
            )
        else:
            suggestions = [
                f"Current: {current_value}, Expected: {expected}",
                "Review the validation requirements",
                "Adjust your input to meet the criteria",
                "Consider using different validation settings",
            ]

        return message, suggestions

    @staticmethod
    def _get_length_suggestions(issue: str, current_value: Any, expected: str) -> List[str]:
        """Get length-specific suggestions."""
        suggestions = []

        if "too short" in issue.lower():
            if isinstance(current_value, int):
                match = re.search(r"minimum:\s*(\d+)", expected)
                if match:
                    min_length = int(match.group(1))
                    deficit = min_length - current_value
                    suggestions.append(f"Add approximately {deficit} more characters/words")
            suggestions.extend(
                [
                    "Expand your content with more details",
                    "Add examples or explanations",
                    "Include additional relevant information",
                ]
            )
        elif "too long" in issue.lower():
            if isinstance(current_value, int):
                match = re.search(r"maximum:\s*(\d+)", expected)
                if match:
                    max_length = int(match.group(1))
                    excess = current_value - max_length
                    suggestions.append(f"Remove approximately {excess} characters/words")
            suggestions.extend(
                [
                    "Trim unnecessary words or phrases",
                    "Focus on the most important points",
                    "Break content into smaller sections",
                ]
            )

        return suggestions

    @staticmethod
    def _get_content_suggestions(issue: str, current_value: Any, expected: str) -> List[str]:
        """Get content-specific suggestions."""
        suggestions = []

        if "missing required" in issue.lower():
            suggestions.extend(
                [
                    f"Include the required content: {expected}",
                    "Make sure all required topics are covered",
                    "Review the content requirements carefully",
                ]
            )
        elif "prohibited content" in issue.lower():
            suggestions.extend(
                [
                    f"Remove or rephrase the prohibited content",
                    "Use alternative wording to express the same idea",
                    "Review content guidelines and restrictions",
                ]
            )

        return suggestions

    @staticmethod
    def _get_format_suggestions(issue: str, current_value: Any, expected: str) -> List[str]:
        """Get format-specific suggestions."""
        suggestions = []

        if "json" in issue.lower():
            suggestions.extend(
                [
                    "Check JSON syntax for missing commas, brackets, or quotes",
                    "Use a JSON validator to identify syntax errors",
                    "Ensure all strings are properly quoted",
                ]
            )
        elif "markdown" in issue.lower():
            suggestions.extend(
                [
                    "Check Markdown syntax for headers, links, and formatting",
                    "Ensure proper spacing around headers and lists",
                    "Validate Markdown with a preview tool",
                ]
            )

        return suggestions


class SifakaError(Exception):
    """Base exception for all Sifaka errors.

    Provides common functionality for error context, suggestions,
    and structured error information.

    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error identifier
        context: Additional context information
        suggestions: List of actionable suggestions
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
    ):
        """Initialize Sifaka error.

        Args:
            message: Human-readable error description
            error_code: Machine-readable error identifier
            context: Additional context information
            suggestions: List of actionable suggestions for resolution
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.context = context or {}
        self.suggestions = suggestions or []

    def __str__(self) -> str:
        """Format error with context and suggestions."""
        parts = [f"{self.error_code}: {self.message}"]

        if self.context:
            parts.append("Context:")
            for key, value in self.context.items():
                parts.append(f"  {key}: {value}")

        if self.suggestions:
            parts.append("Suggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  - {suggestion}")

        return "\n".join(parts)


class ValidationError(SifakaError):
    """Raised when validation fails.

    This error occurs when content fails to meet validation criteria,
    such as length requirements, format constraints, or content policies.
    """

    def __init__(
        self,
        message: str,
        validator_name: Optional[str] = None,
        validation_details: Optional[Dict[str, Any]] = None,
        current_value: Optional[Any] = None,
        expected_value: Optional[str] = None,
        **kwargs,
    ):
        """Initialize validation error with enhanced user-friendly messaging.

        Args:
            message: Human-readable error description
            validator_name: Name of the validator that failed
            validation_details: Detailed validation results
            current_value: Current value that failed validation
            expected_value: Description of what was expected
            **kwargs: Additional arguments passed to SifakaError
        """
        context = kwargs.get("context", {})
        if validator_name:
            context["validator"] = validator_name
        if validation_details:
            context["validation_details"] = validation_details
        if current_value is not None:
            context["current_value"] = current_value
        if expected_value:
            context["expected_value"] = expected_value

        suggestions = kwargs.get("suggestions", [])

        # Use enhanced error message builder if we have enough context
        if not suggestions and validator_name and current_value is not None and expected_value:
            try:
                enhanced_message, enhanced_suggestions = (
                    ErrorMessageBuilder.format_validation_error(
                        validator_name, message, current_value, expected_value
                    )
                )
                # Use enhanced message if it's more descriptive
                if len(enhanced_message) > len(message):
                    message = enhanced_message
                suggestions = enhanced_suggestions
            except Exception:
                # Fall back to default suggestions if enhancement fails
                pass

        # Default suggestions if none provided
        if not suggestions:
            suggestions = [
                "Check input parameters against validation requirements",
                "Review validation configuration and thresholds",
                "Ensure content meets specified criteria",
            ]

        super().__init__(
            message,
            error_code=kwargs.get("error_code", "VALIDATION_ERROR"),
            context=context,
            suggestions=suggestions,
            **{
                k: v for k, v in kwargs.items() if k not in ["context", "suggestions", "error_code"]
            },
        )


class CritiqueError(SifakaError):
    """Raised when critique execution fails.

    This error occurs when critics fail to execute properly,
    such as model API failures, timeout issues, or invalid responses.
    """

    def __init__(
        self,
        message: str,
        critic_name: Optional[str] = None,
        model_error: Optional[Exception] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize critique error with enhanced model-specific messaging.

        Args:
            message: Human-readable error description
            critic_name: Name of the critic that failed
            model_error: Underlying model/API error if applicable
            model_name: Name of the model that failed
            **kwargs: Additional arguments passed to SifakaError
        """
        context = kwargs.get("context", {})
        if critic_name:
            context["critic"] = critic_name
        if model_name:
            context["model"] = model_name
        if model_error:
            context["underlying_error"] = str(model_error)
            context["error_type"] = type(model_error).__name__

        suggestions = kwargs.get("suggestions", [])

        # Use enhanced error message builder for common model errors
        if not suggestions and model_error and model_name:
            try:
                error_str = str(model_error).lower()
                if "api" in error_str and "key" in error_str:
                    # API key error
                    provider = self._extract_provider_from_model(model_name)
                    key_name = f"{provider.upper()}_API_KEY"
                    enhanced_message, enhanced_suggestions = (
                        ErrorMessageBuilder.format_api_key_error(provider, key_name)
                    )
                    message = f"{message}: {enhanced_message}"
                    suggestions = enhanced_suggestions
                elif "rate limit" in error_str or "quota" in error_str:
                    # Rate limit error
                    enhanced_message, enhanced_suggestions = ErrorMessageBuilder.format_model_error(
                        model_name, "rate_limit"
                    )
                    message = f"{message}: {enhanced_message}"
                    suggestions = enhanced_suggestions
                elif "connection" in error_str or "network" in error_str:
                    # Connection error
                    enhanced_message, enhanced_suggestions = ErrorMessageBuilder.format_model_error(
                        model_name, "connection"
                    )
                    message = f"{message}: {enhanced_message}"
                    suggestions = enhanced_suggestions
                elif "model" in error_str and ("not found" in error_str or "invalid" in error_str):
                    # Invalid model error
                    enhanced_message, enhanced_suggestions = ErrorMessageBuilder.format_model_error(
                        model_name, "invalid_model"
                    )
                    message = f"{message}: {enhanced_message}"
                    suggestions = enhanced_suggestions
            except Exception:
                # Fall back to default suggestions if enhancement fails
                pass

        # Default suggestions if none provided
        if not suggestions:
            suggestions = [
                "Check model API credentials and connectivity",
                "Verify critic configuration and parameters",
                "Consider using a different critic or model",
                "Review rate limits and timeout settings",
            ]

        super().__init__(
            message,
            error_code="CRITIQUE_ERROR",
            context=context,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ["context", "suggestions"]},
        )

    @staticmethod
    def _extract_provider_from_model(model_name: str) -> str:
        """Extract provider name from model string."""
        if ":" in model_name:
            provider = model_name.split(":")[0]
            if provider == "openai":
                return "OpenAI"
            elif provider == "anthropic":
                return "Anthropic"
            elif provider == "google" or provider == "gemini":
                return "Google"
            elif provider == "groq":
                return "Groq"
        return "Model Provider"


class GraphExecutionError(SifakaError):
    """Raised when graph execution fails.

    This error occurs when the PydanticAI graph workflow encounters
    issues during execution, such as node failures or state corruption.
    """

    def __init__(
        self,
        message: str,
        node_name: Optional[str] = None,
        execution_stage: Optional[str] = None,
        **kwargs,
    ):
        """Initialize graph execution error.

        Args:
            message: Human-readable error description
            node_name: Name of the graph node that failed
            execution_stage: Stage of execution where failure occurred
            **kwargs: Additional arguments passed to SifakaError
        """
        context = kwargs.get("context", {})
        if node_name:
            context["node"] = node_name
        if execution_stage:
            context["stage"] = execution_stage

        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                "Check graph configuration and node dependencies",
                "Verify state consistency and data flow",
                "Review logs for detailed error information",
                "Consider simplifying the workflow",
            ]

        super().__init__(
            message,
            error_code="GRAPH_EXECUTION_ERROR",
            context=context,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ["context", "suggestions"]},
        )


class ConfigurationError(SifakaError):
    """Raised when configuration is invalid.

    This error occurs when Sifaka configuration is missing, invalid,
    or incompatible with the current environment or requirements.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize configuration error with enhanced guidance.

        Args:
            message: Human-readable error description
            config_key: Configuration key that caused the error
            config_value: Invalid configuration value
            **kwargs: Additional arguments passed to SifakaError
        """
        context = kwargs.get("context", {})
        if config_key:
            context["config_key"] = config_key
        if config_value is not None:
            context["config_value"] = config_value

        suggestions = kwargs.get("suggestions", [])

        # Provide specific suggestions based on configuration key
        if not suggestions and config_key:
            suggestions = self._get_config_specific_suggestions(config_key, config_value, message)

        # Default suggestions if none provided
        if not suggestions:
            suggestions = [
                "Check configuration file syntax and values",
                "Verify environment variables are set correctly",
                "Review documentation for valid configuration options",
                "Use SifakaConfig.simple() for basic setup",
            ]

        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            context=context,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ["context", "suggestions"]},
        )

    @staticmethod
    def _get_config_specific_suggestions(
        config_key: str, config_value: Any, message: str
    ) -> List[str]:
        """Get configuration-specific suggestions based on the key and error."""
        suggestions = []

        if "max_iterations" in config_key:
            suggestions.extend(
                [
                    "Set max_iterations to a positive integer (1-20)",
                    "Example: max_iterations=3",
                    "Higher values allow more improvement cycles but take longer",
                ]
            )
        elif "min_length" in config_key or "max_length" in config_key:
            suggestions.extend(
                [
                    "Set length values to positive integers",
                    "Ensure min_length <= max_length if both are specified",
                    "Example: min_length=50, max_length=500",
                ]
            )
        elif "model" in config_key:
            suggestions.extend(
                [
                    "Use format 'provider:model-name' (e.g., 'openai:gpt-4')",
                    "Supported providers: openai, anthropic, google, groq",
                    "Verify the model name is correct and available",
                    "Check that you have API access to the model",
                ]
            )
        elif "critics" in config_key:
            suggestions.extend(
                [
                    "Use a list of critic names: ['reflexion', 'constitutional']",
                    "Available critics: reflexion, constitutional, self_refine",
                    "Example: critics=['reflexion', 'constitutional']",
                ]
            )
        elif "api_key" in config_key.lower() or "key" in config_key.lower():
            suggestions.extend(
                [
                    "Set your API key as an environment variable",
                    "Example: export OPENAI_API_KEY='your-key-here'",
                    "Get API keys from the respective provider websites",
                    "Ensure the API key has necessary permissions",
                ]
            )
        else:
            # Generic suggestions for unknown config keys
            suggestions.extend(
                [
                    f"Check the '{config_key}' configuration value",
                    "Review the documentation for valid options",
                    "Use SifakaConfig.simple() with valid parameters",
                ]
            )

        return suggestions
