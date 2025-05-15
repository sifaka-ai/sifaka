"""
Chain interfaces for Sifaka.

This module defines the interfaces for chains in the Sifaka framework.
These interfaces establish a common contract for chain behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Chain**: Base interface for all chains
   - **PromptManager**: Interface for prompt managers
   - **ValidationManager**: Interface for validation managers
   - **RetryStrategy**: Interface for retry strategies
   - **ResultFormatter**: Interface for result formatters
   - **Model**: Interface for text generation models
   - **Validator**: Interface for output validators
   - **Improver**: Interface for output improvers
   - **ChainFormatter**: Interface for result formatters
   - **ChainComponent**: Base interface for all chain components
   - **ChainPlugin**: Interface for chain plugins

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.

## State Management

The interfaces support standardized state management:
- Single _state_manager attribute for all mutable state
- State initialization during construction
- State access through state manager methods
- Clear separation of configuration and state

## Error Handling

The interfaces define error handling patterns:
- ValueError for invalid inputs
- RuntimeError for execution failures
- TypeError for type mismatches
- ModelError: Raised when text generation fails
- ValidationError: Raised when validation fails
- ImproverError: Raised when improvement fails
- FormatterError: Raised when formatting fails
- Detailed error tracking and reporting

## Execution Tracking

The interfaces support execution tracking:
- Execution count tracking
- Execution time tracking
- Success/failure tracking
- Performance statistics
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
    TYPE_CHECKING,
)

# Type exports for type checking
ChainType = TypeVar("ChainType")
ResultType = TypeVar("ResultType")
RetryStrategyType = TypeVar("RetryStrategyType")
ValidationManagerType = TypeVar("ValidationManagerType")
ResultFormatterType = TypeVar("ResultFormatterType")
PromptManagerType = TypeVar("PromptManagerType")
ModelType = TypeVar("ModelType")
ValidatorType = TypeVar("ValidatorType")
ImproverType = TypeVar("ImproverType")
FormatterType = TypeVar("FormatterType")
PluginType = TypeVar("PluginType")

# For type checking purposes, define placeholder objects
if TYPE_CHECKING:

    class Model(Protocol):
        def generate(self, prompt: str, **kwargs: Any) -> str: ...
        @property
        def name(self) -> str: ...

    class Validator(Protocol):
        def validate(self, text: str, **kwargs: Any) -> "ValidationResult": ...
        @property
        def name(self) -> str: ...

    class Improver(Protocol):
        def improve(self, text: str, feedback: Optional[str] = None, **kwargs: Any) -> str: ...
        @property
        def name(self) -> str: ...

    class ChainFormatter(Protocol):
        def format(self, text: str, **kwargs: Any) -> str: ...
        @property
        def name(self) -> str: ...

    class ChainComponent(Protocol):
        @property
        def name(self) -> str: ...
        @property
        def description(self) -> str: ...

    class ValidationResult:
        passed: bool
        message: str
        score: Optional[float] = None
        issues: List[str] = []
        suggestions: List[str] = []
        metadata: Dict[str, Any] = {}

    class Chain(Protocol):
        def run(self, input_text: str, **kwargs: Any) -> str: ...
        @property
        def name(self) -> str: ...

    class ChainPlugin(Protocol):
        def initialize(self) -> None: ...
        @property
        def name(self) -> str: ...

    class PromptManager(Protocol):
        def create_prompt(self, template: str, variables: Dict[str, Any], **kwargs: Any) -> str: ...
        @property
        def name(self) -> str: ...

    class ValidationManager(Protocol):
        def validate(self, text: str, **kwargs: Any) -> ValidationResult: ...
        @property
        def name(self) -> str: ...

    class RetryStrategy(Protocol):
        def should_retry(self, attempt: int, result: Any, **kwargs: Any) -> bool: ...
        @property
        def name(self) -> str: ...

    class ResultFormatter(Protocol):
        def format_result(self, result: Any, **kwargs: Any) -> Any: ...
        @property
        def name(self) -> str: ...

else:
    # These imports are only used at runtime, not during type checking
    try:
        from .model import Model
    except ImportError:
        Model = None

    try:
        from .validator import Validator
    except ImportError:
        Validator = None

    try:
        from .improver import Improver
    except ImportError:
        Improver = None

    try:
        from .formatter import ChainFormatter
    except ImportError:
        ChainFormatter = None

    try:
        from .base import ChainComponent
    except ImportError:
        ChainComponent = None

    try:
        from .results import ValidationResult
    except ImportError:
        ValidationResult = None

    try:
        from .chain import Chain
    except ImportError:
        Chain = None

    try:
        from .plugin import ChainPlugin
    except ImportError:
        ChainPlugin = None

    try:
        from .manager import PromptManager, ValidationManager
    except ImportError:
        PromptManager = None
        ValidationManager = None

    try:
        from .retry_strategy import RetryStrategy
    except ImportError:
        RetryStrategy = None

    try:
        from .result_formatter import ResultFormatter
    except ImportError:
        ResultFormatter = None

# Define exports
__all__ = [
    "Chain",
    "Model",
    "Validator",
    "Improver",
    "ChainFormatter",
    "ChainComponent",
    "ValidationResult",
    "ChainPlugin",
    "PromptManager",
    "ValidationManager",
    "RetryStrategy",
    "ResultFormatter",
]
