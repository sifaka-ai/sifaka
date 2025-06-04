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

from typing import Any, Dict, List, Optional


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
        **kwargs
    ):
        """Initialize validation error.
        
        Args:
            message: Human-readable error description
            validator_name: Name of the validator that failed
            validation_details: Detailed validation results
            **kwargs: Additional arguments passed to SifakaError
        """
        context = kwargs.get("context", {})
        if validator_name:
            context["validator"] = validator_name
        if validation_details:
            context["validation_details"] = validation_details
        
        suggestions = kwargs.get("suggestions", [])
        if not suggestions:
            suggestions = [
                "Check input parameters against validation requirements",
                "Review validation configuration and thresholds",
                "Ensure content meets specified criteria",
            ]
        
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            context=context,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ["context", "suggestions"]}
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
        **kwargs
    ):
        """Initialize critique error.
        
        Args:
            message: Human-readable error description
            critic_name: Name of the critic that failed
            model_error: Underlying model/API error if applicable
            **kwargs: Additional arguments passed to SifakaError
        """
        context = kwargs.get("context", {})
        if critic_name:
            context["critic"] = critic_name
        if model_error:
            context["underlying_error"] = str(model_error)
            context["error_type"] = type(model_error).__name__
        
        suggestions = kwargs.get("suggestions", [])
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
            **{k: v for k, v in kwargs.items() if k not in ["context", "suggestions"]}
        )


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
        **kwargs
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
            **{k: v for k, v in kwargs.items() if k not in ["context", "suggestions"]}
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
        **kwargs
    ):
        """Initialize configuration error.
        
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
        if not suggestions:
            suggestions = [
                "Check configuration file syntax and values",
                "Verify environment variables are set correctly",
                "Review documentation for valid configuration options",
                "Use SifakaConfig.create_default() for basic setup",
            ]
        
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            context=context,
            suggestions=suggestions,
            **{k: v for k, v in kwargs.items() if k not in ["context", "suggestions"]}
        )
