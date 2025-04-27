"""
Base classes for Sifaka rules.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, Any, Optional, Callable, Union, Tuple
import hashlib

from pydantic import BaseModel, Field, ConfigDict, field_validator


class RuleResult(BaseModel):
    """
    Result of a rule validation.

    Attributes:
        passed: Whether the validation passed
        message: Message explaining the result
        metadata: Additional metadata about the result
        score: Optional confidence score between 0 and 1
    """

    passed: bool = Field(..., description="Whether the validation passed")
    message: str = Field(..., description="Message explaining the result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    score: Optional[float] = Field(
        default=None, ge=0, le=1, description="Confidence score between 0 and 1"
    )

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

    name: str = Field(..., description="Name of the rule")
    description: str = Field(..., description="Description of the rule")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")
    cache_size: int = Field(default=0, ge=0, description="Size of the validation cache")
    priority: int = Field(default=1, description="Priority of the rule (higher runs first)")
    cost: int = Field(default=1, ge=0, description="Cost of running the rule")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("priority")
    def validate_priority(cls, v: int) -> int:
        """Validate priority is positive."""
        if v < 0:
            raise ValueError("Priority must be non-negative")
        return v

    @field_validator("cost")
    def validate_cost(cls, v: int) -> int:
        """Validate cost is positive."""
        if v < 0:
            raise ValueError("Cost must be non-negative")
        return v

    @field_validator("cache_size")
    def validate_cache_size(cls, v: int) -> int:
        """Validate cache_size is non-negative."""
        if v < 0:
            raise ValueError("Cache size must be non-negative")
        return v

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        cache_size: int = 0,
        priority: int = 1,
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

    def _get_cache_key(self, output: str, **kwargs) -> str:
        """
        Generate a cache key for the output.

        Args:
            output: The output to validate
            **kwargs: Additional validation context

        Returns:
            A cache key string
        """
        # Create a hash of the output and config
        hasher = hashlib.md5()
        hasher.update(output.encode())
        hasher.update(str(self.config).encode())
        return hasher.hexdigest()

    @abstractmethod
    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Implement the validation logic.

        Args:
            output: The output to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results
        """
        pass

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate an output.

        This method handles input validation and caching if enabled.

        Args:
            output: The output to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results

        Raises:
            ValueError: If output is None or not a string
        """
        if output is None:
            raise ValueError("Output cannot be None")
        if not isinstance(output, str):
            raise ValueError("Output must be a string")

        # Check cache if enabled
        if self.cache_size > 0:
            cache_key = self._get_cache_key(output, **kwargs)
            if cache_key in self._result_cache:
                return self._result_cache[cache_key]

        # Validate output
        result = self._validate_impl(output, **kwargs)

        # Update cache if enabled
        if self.cache_size > 0:
            cache_key = self._get_cache_key(output, **kwargs)
            self._result_cache[cache_key] = result
            if len(self._result_cache) > self.cache_size:
                self._result_cache.popitem(last=False)

        return result


class FunctionRule(Rule):
    """
    A rule that wraps a function.

    This allows for simple rule creation using functions.

    Attributes:
        func: The function to use for validation
        name: The name of the rule
    """

    func: Callable

    def _validate_impl(self, output: str, **kwargs) -> RuleResult:
        """
        Validate using the provided function.

        Args:
            output: The LLM output to validate
            **kwargs: Additional validation context

        Returns:
            The result of the validation
        """
        result = self.func(output, **kwargs)

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
