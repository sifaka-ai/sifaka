"""
Base classes for Sifaka rules.
"""

from typing import Dict, Any, Optional, Callable, Union, Tuple
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


class Rule(BaseModel):
    """
    Base class for all Sifaka rules.

    A rule validates an LLM output against a specific criterion.

    Attributes:
        name: The name of the rule
        description: Description of the rule
        config: Configuration for the rule
    """

    name: str
    description: str
    config: Dict[str, Any] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a rule.

        Args:
            name: The name of the rule
            description: Description of the rule
            config: Configuration for the rule
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            description=description,
            config=config or {},
            **kwargs,
        )

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate the output against this rule.

        Args:
            output: The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            The result of the validation

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement validate()")


class FunctionRule(Rule):
    """
    A rule that wraps a function.

    This allows for simple rule creation using functions.

    Attributes:
        func: The function to use for validation
        name: The name of the rule
    """

    func: Callable

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate the output using the wrapped function.

        Args:
            output: The LLM output to validate
            **kwargs: Additional context for validation

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
