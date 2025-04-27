"""
Base classes for Sifaka rules.
"""

from typing import Dict, Any, Optional, Callable
from pydantic import BaseModel, Field


class RuleResult(BaseModel):
    """
    Result of a rule validation.

    Attributes:
        passed (bool): Whether the validation passed
        message (str): Message explaining the result
        metadata (Dict[str, Any]): Additional metadata about the result
    """

    passed: bool
    message: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Rule(BaseModel):
    """
    Base class for all Sifaka rules.

    A rule validates an LLM output against a specific criterion.

    Attributes:
        name (str): The name of the rule
    """

    name: str

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, name: Optional[str] = None, **data):
        """
        Initialize a rule.

        Args:
            name (Optional[str]): The name of the rule
            **data: Additional data for the rule
        """
        if name is not None:
            data["name"] = name
        elif "name" not in data:
            data["name"] = self.__class__.__name__

        super().__init__(**data)

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate the output against this rule.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation
        """
        raise NotImplementedError("Subclasses must implement validate()")


class FunctionRule(Rule):
    """
    A rule that wraps a function.

    This allows for simple rule creation using functions.

    Attributes:
        func (Callable): The function to use for validation
        name (str): The name of the rule
    """

    func: Callable

    def validate(self, output: str, **kwargs) -> RuleResult:
        """
        Validate the output using the wrapped function.

        Args:
            output (str): The LLM output to validate
            **kwargs: Additional context for validation

        Returns:
            RuleResult: The result of the validation
        """
        result = self.func(output, **kwargs)

        if isinstance(result, RuleResult):
            return result
        elif isinstance(result, bool):
            return RuleResult(passed=result, message="" if result else f"Rule {self.name} failed")
        elif isinstance(result, tuple) and len(result) >= 2:
            passed, message = result[0], result[1]
            metadata = result[2] if len(result) > 2 else {}
            return RuleResult(passed=passed, message=message, metadata=metadata)
        else:
            raise ValueError(
                f"Rule function must return a RuleResult, a bool, or a tuple of (bool, str, dict). "
                f"Got {type(result)}"
            )
