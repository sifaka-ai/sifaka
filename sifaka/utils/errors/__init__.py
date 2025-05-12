"""
Error Handling Module

This module provides standardized error handling utilities for the Sifaka framework,
including exception classes, error handling functions, and logging utilities.

## Overview
The error handling system in Sifaka provides a structured approach to handling
errors across the framework. It includes a comprehensive exception hierarchy,
standardized error handling patterns, and component-specific error handlers.

The system is designed to provide consistent error handling behavior, detailed
error information, and appropriate logging across all components.

## Components
- **Exception Classes**: Structured hierarchy of exception classes
- **Error Handling Functions**: Functions for standardized error handling
- **Error Result Classes**: Models for representing errors in results
- **Component-Specific Error Handlers**: Specialized handlers for different components
- **Safe Execution Functions**: Functions for safely executing operations

## Exception Hierarchy

Sifaka uses a structured exception hierarchy:

1. **SifakaError**: Base class for all Sifaka exceptions
   - **ValidationError**: Raised when validation fails
   - **ConfigurationError**: Raised when configuration is invalid
   - **ProcessingError**: Raised when processing fails
     - **ResourceError**: Raised when a resource is unavailable
     - **TimeoutError**: Raised when an operation times out

2. **Component-Specific Errors**:
   - **ChainError**: Raised by chain components
   - **ModelError**: Raised by model providers
   - **RuleError**: Raised during rule validation
   - **CriticError**: Raised by critics
   - **ClassifierError**: Raised by classifiers
   - **RetrievalError**: Raised during retrieval operations

3. **Common Error Types**:
   - **InputError**: Raised when input is invalid
   - **StateError**: Raised when state is invalid
   - **DependencyError**: Raised when a dependency fails
"""

# Import all public classes and functions from the modules
from .base import (
    SifakaError,
    ValidationError,
    ConfigurationError,
    ProcessingError,
    ResourceError,
    TimeoutError,
    InputError,
    StateError,
    DependencyError,
    InitializationError,
    ComponentError,
)

from .component import (
    ChainError,
    ImproverError,
    FormatterError,
    PluginError,
    ModelError,
    RuleError,
    CriticError,
    ClassifierError,
    RetrievalError,
)

from .handling import (
    handle_error,
    try_operation,
    log_error,
    handle_component_error,
    create_error_handler,
    handle_chain_error,
    handle_model_error,
    handle_rule_error,
    handle_critic_error,
    handle_classifier_error,
    handle_retrieval_error,
)

from .results import (
    ErrorResult,
    create_error_result,
    create_error_result_factory,
    create_chain_error_result,
    create_model_error_result,
    create_rule_error_result,
    create_critic_error_result,
    create_classifier_error_result,
    create_retrieval_error_result,
)

from .safe_execution import (
    try_component_operation,
    safely_execute_component_operation,
    create_safe_execution_factory,
    safely_execute_chain,
    safely_execute_model,
    safely_execute_rule,
    safely_execute_critic,
    safely_execute_classifier,
    safely_execute_retrieval,
    safely_execute_component,
)

from .logging import (
    configure_error_logging,
    get_error_logger,
)

# Define __all__ to specify public API
__all__ = [
    # Base error classes
    "SifakaError",
    "ValidationError",
    "ConfigurationError",
    "ProcessingError",
    "ResourceError",
    "TimeoutError",
    "InputError",
    "StateError",
    "DependencyError",
    "InitializationError",
    "ComponentError",
    # Component-specific error classes
    "ChainError",
    "ImproverError",
    "FormatterError",
    "PluginError",
    "ModelError",
    "RuleError",
    "CriticError",
    "ClassifierError",
    "RetrievalError",
    # Error handling functions
    "handle_error",
    "try_operation",
    "log_error",
    # Error result classes
    "ErrorResult",
    # Component-specific error handlers
    "handle_component_error",
    "create_error_handler",
    "handle_chain_error",
    "handle_model_error",
    "handle_rule_error",
    "handle_critic_error",
    "handle_classifier_error",
    "handle_retrieval_error",
    # Error result creation functions
    "create_error_result",
    "create_error_result_factory",
    "create_chain_error_result",
    "create_model_error_result",
    "create_rule_error_result",
    "create_critic_error_result",
    "create_classifier_error_result",
    "create_retrieval_error_result",
    # Safe execution functions
    "try_component_operation",
    "safely_execute_component_operation",
    "create_safe_execution_factory",
    "safely_execute_chain",
    "safely_execute_model",
    "safely_execute_rule",
    "safely_execute_critic",
    "safely_execute_classifier",
    "safely_execute_retrieval",
    "safely_execute_component",
    # Logging functions
    "configure_error_logging",
    "get_error_logger",
]
