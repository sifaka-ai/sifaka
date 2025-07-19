"""Protocol definitions for Sifaka interfaces.

This module defines Protocol types that specify the expected interfaces
for various components. These provide better type checking than ABCs
while allowing duck typing.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .models import CritiqueResult, SifakaResult, ValidationResult


@runtime_checkable
class CriticProtocol(Protocol):
    """Protocol for critic implementations.

    Any class implementing these methods can be used as a critic,
    regardless of inheritance.
    """

    @property
    def name(self) -> str:
        """Return the critic's name."""
        ...

    async def critique(self, text: str, result: SifakaResult) -> CritiqueResult:
        """Perform critique on the text."""
        ...


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol for validator implementations.

    Any class implementing these methods can be used as a validator,
    regardless of inheritance.
    """

    @property
    def name(self) -> str:
        """Return the validator's name."""
        ...

    def validate(self, text: str) -> ValidationResult:
        """Validate the text."""
        ...


@runtime_checkable
class StorageProtocol(Protocol):
    """Protocol for storage backend implementations.

    Any class implementing these methods can be used as storage,
    regardless of inheritance.
    """

    async def save(self, result_id: str, result: SifakaResult) -> bool:
        """Save a result."""
        ...

    async def load(self, result_id: str) -> Optional[SifakaResult]:
        """Load a result by ID."""
        ...

    async def delete(self, result_id: str) -> bool:
        """Delete a result."""
        ...

    async def exists(self, result_id: str) -> bool:
        """Check if a result exists."""
        ...

    async def list(self, limit: int = 100, offset: int = 0) -> List[str]:
        """List stored result IDs."""
        ...


@runtime_checkable
class ToolProtocol(Protocol):
    """Protocol for tool implementations.

    Any class implementing these methods can be used as a tool,
    regardless of inheritance.
    """

    @property
    def name(self) -> str:
        """Return the tool's name."""
        ...

    @property
    def description(self) -> str:
        """Return the tool's description."""
        ...

    async def __call__(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Execute the tool with a query.

        See ToolCallParams in core.type_defs for expected kwargs fields.
        """
        ...


@runtime_checkable
class MiddlewareProtocol(Protocol):
    """Protocol for middleware implementations.

    Any class implementing these methods can be used as middleware,
    regardless of inheritance.
    """

    @property
    def name(self) -> str:
        """Return the middleware's name."""
        ...

    async def process_request(
        self, text: str, config: Any, context: Dict[str, Any]
    ) -> tuple[str, Any, Dict[str, Any]]:
        """Process before improvement."""
        ...

    async def process_response(
        self, result: SifakaResult, context: Dict[str, Any]
    ) -> SifakaResult:
        """Process after improvement."""
        ...


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM client implementations.

    Any class implementing these methods can be used as an LLM client,
    regardless of inheritance.
    """

    @property
    def model(self) -> str:
        """Return the model name."""
        ...

    @property
    def temperature(self) -> float:
        """Return the temperature setting."""
        ...

    async def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """Complete a conversation.

        See LLMCompleteParams in core.type_defs for expected kwargs fields.
        """
        ...


# Type guard functions
def is_critic(obj: Any) -> bool:
    """Check if an object implements the CriticProtocol."""
    return isinstance(obj, CriticProtocol)


def is_validator(obj: Any) -> bool:
    """Check if an object implements the ValidatorProtocol."""
    return isinstance(obj, ValidatorProtocol)


def is_storage(obj: Any) -> bool:
    """Check if an object implements the StorageProtocol."""
    return isinstance(obj, StorageProtocol)


def is_tool(obj: Any) -> bool:
    """Check if an object implements the ToolProtocol."""
    return isinstance(obj, ToolProtocol)


def is_middleware(obj: Any) -> bool:
    """Check if an object implements the MiddlewareProtocol."""
    return isinstance(obj, MiddlewareProtocol)


__all__ = [
    "CriticProtocol",
    "ValidatorProtocol",
    "StorageProtocol",
    "ToolProtocol",
    "MiddlewareProtocol",
    "LLMClientProtocol",
    "is_critic",
    "is_validator",
    "is_storage",
    "is_tool",
    "is_middleware",
]
