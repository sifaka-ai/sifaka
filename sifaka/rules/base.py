"""
Base classes for Sifaka rules.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, Any, Optional, Callable, Union, Tuple
import hashlib

from pydantic import BaseModel, Field, ConfigDict


class RuleResult(BaseModel):
    """
    Result of a rule validation.

    Attributes:
        passed: Whether the validation passed
        message: Message explaining the result
        metadata: Additional metadata about the result
    """

    passed: bool
    message: Optional[str] = None
    metadata: Dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def failed(self) -> bool:
        """Return whether the validation failed."""
        return not self.passed

    def __bool__(self) -> bool:
        """Return whether the validation passed."""
        return self.passed


class Rule(ABC, BaseModel):
    """
    Base class for all Sifaka rules.

    A rule validates an LLM output against a specific criterion.

    Attributes:
        name: The name of the rule
        description: Description of the rule
        config: Configuration for the rule
        cache_size: Size of the LRU cache (0 to disable)
        priority: Priority of the rule (higher numbers run first)
        cost: Estimated computational cost (higher numbers are more expensive)
    """

    name: str
    description: str
    config: Dict[str, Any] = {}
    cache_size: int = 0
    priority: int = 0
    cost: int = 1

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        cache_size: int = 0,
        priority: int = 0,
        cost: int = 1,
        **kwargs,
    ) -> None:
        """
        Initialize a rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration for the rule
            cache_size: Size of the LRU cache (0 to disable)
            priority: Priority of the rule (higher numbers run first)
            cost: Estimated computational cost (higher numbers are more expensive)
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            description=description,
            config=config or {},
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            **kwargs,
        )

        # Initialize cache if enabled
        if self.cache_size > 0:
            self._cached_validate = lru_cache(maxsize=self.cache_size)(self._validate_impl)
        else:
            self._cached_validate = self._validate_impl

    def _get_cache_key(self, output: str) -> str:
        """
        Generate a cache key for the output.

        Args:
            output: The output to validate

        Returns:
            A cache key string
        """
        # Create a hash of the output and config
        hasher = hashlib.md5()
        hasher.update(output.encode())
        hasher.update(str(self.config).encode())
        return hasher.hexdigest()

    @abstractmethod
    def _validate_impl(self, output: str) -> RuleResult:
        """
        Implement the validation logic.

        Args:
            output: The output to validate

        Returns:
            RuleResult with validation results
        """
        pass

    def validate(self, output: str) -> RuleResult:
        """
        Validate an output.

        This method handles caching if enabled.

        Args:
            output: The output to validate

        Returns:
            RuleResult with validation results
        """
        if self.cache_size > 0:
            # Use cache key for cached validation
            cache_key = self._get_cache_key(output)
            return self._cached_validate(cache_key)
        else:
            # Direct validation without caching
            return self._validate_impl(output)


class FunctionRule(Rule):
    """
    A rule that wraps a function.

    This allows for simple rule creation using functions.

    Attributes:
        func: The function to use for validation
        name: The name of the rule
    """

    func: Callable

    def _validate_impl(self, output: str) -> RuleResult:
        """
        Validate using the provided function.

        Args:
            output: The LLM output to validate

        Returns:
            The result of the validation
        """
        result = self.func(output)

        if isinstance(result, RuleResult):
            return result
        if isinstance(result, bool):
            return RuleResult(passed=result, message="" if result else f"Rule {self.name} failed")
        if isinstance(result, tuple) and len(result) >= 2:
            passed, message = result[0], result[1]
            metadata = result[2] if len(result) > 2 else {}
            return RuleResult(passed=passed, message=message, metadata=metadata)

        raise ValueError(
            f"Function {self.func.__name__} must return a bool, RuleResult, "
            f"or tuple of (bool, str[, dict])"
        )
