"""Utility functions for the thought module."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from sifaka.core.thought.constants import (
    DEFAULT_CHAIN_ID,
    THOUGHT_KEY_PREFIX,
    THOUGHT_KEY_SEPARATOR,
)

if TYPE_CHECKING:
    from sifaka.core.thought.thought import Thought


def generate_thought_key(chain_id: Optional[str], iteration: int) -> str:
    """Generate a storage key for a thought.

    Args:
        chain_id: The chain ID (uses default if None).
        iteration: The iteration number.

    Returns:
        The storage key in format "thought_{chain_id}_{iteration}".
    """
    effective_chain_id = chain_id or DEFAULT_CHAIN_ID
    return f"{THOUGHT_KEY_PREFIX}{THOUGHT_KEY_SEPARATOR}{effective_chain_id}{THOUGHT_KEY_SEPARATOR}{iteration}"


def parse_thought_key(key: str) -> Optional[tuple[str, int]]:
    """Parse a thought key to extract chain_id and iteration.

    Args:
        key: The storage key to parse.

    Returns:
        Tuple of (chain_id, iteration) if valid, None otherwise.
    """
    if not key.startswith(f"{THOUGHT_KEY_PREFIX}{THOUGHT_KEY_SEPARATOR}"):
        return None

    parts = key.split(THOUGHT_KEY_SEPARATOR)
    if len(parts) < 3:
        return None

    try:
        iteration = int(parts[-1])
        chain_id = THOUGHT_KEY_SEPARATOR.join(parts[1:-1])
        return chain_id, iteration
    except ValueError:
        return None


def parse_timestamp(timestamp_data: Any) -> datetime:
    """Parse timestamp data into a datetime object.

    Args:
        timestamp_data: String timestamp or datetime object.

    Returns:
        Parsed datetime object, or current time if parsing fails.
    """
    if isinstance(timestamp_data, datetime):
        return timestamp_data

    if isinstance(timestamp_data, str):
        try:
            return datetime.fromisoformat(timestamp_data)
        except ValueError:
            pass

    # Fallback to current time
    return datetime.now()


def create_thought_summary(thought: "Thought") -> str:
    """Create a summary string for a thought.

    Args:
        thought: The thought to summarize.

    Returns:
        A brief summary string.
    """
    text_length = len(thought.text or "")
    validation_count = len(thought.validation_results or {})
    feedback_count = len(thought.critic_feedback or [])

    return (
        f"Iteration {thought.iteration}: {text_length} chars, "
        f"{validation_count} validations, {feedback_count} feedback"
    )


def extract_chain_ids_from_keys(keys: list[str]) -> set[str]:
    """Extract unique chain IDs from a list of thought keys.

    Args:
        keys: List of storage keys.

    Returns:
        Set of unique chain IDs.
    """
    chain_ids = set()
    for key in keys:
        parsed = parse_thought_key(key)
        if parsed:
            chain_id, _ = parsed
            chain_ids.add(chain_id)
    return chain_ids


def safe_operation(operation_name: str, logger):
    """Decorator for safe storage operations with consistent error handling.

    Args:
        operation_name: Name of the operation for logging.
        logger: Logger instance to use.

    Returns:
        Decorator function.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to {operation_name}: {e}")
                # Return appropriate default based on function return type
                import inspect
                from typing import get_args, get_origin

                sig = inspect.signature(func)
                return_annotation = sig.return_annotation

                # Handle Optional types
                if hasattr(return_annotation, "__origin__"):
                    origin = get_origin(return_annotation)
                    if origin is Union:
                        args = get_args(return_annotation)
                        if len(args) == 2 and type(None) in args:
                            # This is Optional[T], return None
                            return None

                # Handle basic types
                if return_annotation == bool:
                    return False
                elif return_annotation == int:
                    return 0
                elif return_annotation == list or (
                    hasattr(return_annotation, "__origin__")
                    and get_origin(return_annotation) is list
                ):
                    return []
                elif (
                    return_annotation == dict
                    or return_annotation == Dict
                    or (
                        hasattr(return_annotation, "__origin__")
                        and get_origin(return_annotation) is dict
                    )
                ):
                    return {}
                else:
                    return None

        return wrapper

    return decorator
