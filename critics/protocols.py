"""Protocol definitions for the critic system."""

from typing import Dict, Any, Protocol, runtime_checkable, TypeVar, Union, List
from typing_extensions import TypeGuard

T = TypeVar("T")
ConfigDict = Dict[str, Union[str, int, float, bool, List[T], Dict[str, T]]]


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for language model providers."""

    def invoke(self, prompt: str) -> Dict[str, Any]: ...


@runtime_checkable
class TextValidator(Protocol):
    """Protocol for text validation."""

    def validate(self, text: str) -> bool: ...


@runtime_checkable
class TextCritic(Protocol):
    """Protocol for text critique."""

    def critique(self, text: str) -> Dict[str, Any]: ...


@runtime_checkable
class TextImprover(Protocol):
    """Protocol for text improvement."""

    def improve(self, text: str, feedback: str) -> str: ...


@runtime_checkable
class PromptFactory(Protocol):
    """Protocol for prompt generation."""

    def create_critique_prompt(self, text: str) -> str: ...
    def create_validation_prompt(self, text: str) -> str: ...
    def create_improvement_prompt(self, text: str, feedback: str) -> str: ...


def is_config_dict(v: Any) -> TypeGuard[ConfigDict]:
    """Validate and narrow type of configuration dictionary.

    Args:
        v: Value to check

    Returns:
        bool: True if value is a valid config dictionary
    """
    if v is None:
        return True
    if not isinstance(v, dict):
        return False
    return all(
        isinstance(k, str) and isinstance(v, (str, int, float, bool, list, dict))
        for k, v in v.items()
    )
