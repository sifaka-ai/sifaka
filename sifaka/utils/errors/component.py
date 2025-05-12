"""
Component-Specific Error Classes

This module defines component-specific error classes for the Sifaka framework.
These classes extend the base SifakaError class and provide specialized error
handling for different component types.

## Classes
- **ChainError**: Raised by chain components
- **ImproverError**: Raised when improver refinement fails
- **FormatterError**: Raised when formatting fails
- **PluginError**: Raised when a plugin operation fails
- **ModelError**: Raised by model providers
- **RuleError**: Raised during rule validation
- **CriticError**: Raised by critics
- **ClassifierError**: Raised by classifiers
- **RetrievalError**: Raised during retrieval operations
"""

from .base import SifakaError, ValidationError


class ChainError(SifakaError):
    """Error raised by chain components.

    This error is raised when a chain component encounters an error,
    such as during orchestration, execution, or result processing.

    ## Architecture
    ChainError is a specialized SifakaError for chain-related errors.
    It inherits all functionality from SifakaError and serves as the
    base class for more specific chain-related errors.

    ## Examples
    ```python
    # Raising a ChainError
    raise ChainError(
        "Chain execution failed",
        metadata={"chain_id": "text_generation", "step": "model_call"}
    )

    # Catching chain-specific errors
    try:
        result = (chain and chain.run(prompt)
    except ChainError as e:
        print(f"Chain error: {e.message}")
        print(f"Chain error metadata: {e.metadata}")
    except SifakaError as e:
        print(f"Other Sifaka error: {e.message}")
    ```

    Attributes:
        message (str): Human-readable error message
        metadata (Dict[str, Any]): Additional error context and details
    """

    pass


class ImproverError(ChainError):
    """Error raised when improver refinement fails.

    This error is raised when an improver component fails to refine output,
    such as during critic-based improvement or other refinement processes.
    """

    pass


class FormatterError(ChainError):
    """Error raised when result formatting fails.

    This error is raised when a formatter component fails to format results,
    such as during output formatting or structure conversion.
    """

    pass


class PluginError(ChainError):
    """Error raised when plugin operations fail.

    This error is raised when a plugin encounters an error during execution,
    such as during initialization, processing, or cleanup.
    """

    pass


class ModelError(SifakaError):
    """Error raised by model providers.

    This error is raised when a model provider encounters an error,
    such as during model initialization, inference, or API communication.
    """

    pass


class RuleError(ValidationError):
    """Error raised during rule validation.

    This error is raised when a rule encounters an error during validation,
    such as when a rule's validation logic fails or produces unexpected results.
    """

    pass


class CriticError(SifakaError):
    """Error raised by critics.

    This error is raised when a critic encounters an error,
    such as during critique generation, feedback processing, or improvement.
    """

    pass


class ClassifierError(SifakaError):
    """Error raised by classifiers.

    This error is raised when a classifier encounters an error,
    such as during classification, model inference, or result processing.
    """

    pass


class RetrievalError(SifakaError):
    """Error raised during retrieval operations.

    This error is raised when a retrieval operation encounters an error,
    such as during document retrieval, indexing, or query processing.
    """

    pass
