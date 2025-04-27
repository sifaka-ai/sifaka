"""
Base classes for Sifaka critiques.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class Critique(BaseModel):
    """
    Base class for all Sifaka critiques.

    A critique improves an LLM output based on rule violations or other criteria.

    Attributes:
        name (str): The name of the critique
    """

    name: str

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, name: Optional[str] = None, **data):
        """
        Initialize a critique.

        Args:
            name (Optional[str]): The name of the critique
            **data: Additional data for the critique
        """
        if name is not None:
            data["name"] = name
        elif "name" not in data:
            data["name"] = self.__class__.__name__

        super().__init__(**data)

    def improve(
        self,
        output: str,
        prompt: Optional[str] = None,
        rule_violations: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """
        Improve the output based on rule violations or other criteria.

        Args:
            output (str): The LLM output to improve
            prompt (Optional[str]): The original prompt that generated the output
            rule_violations (Optional[List[Dict[str, Any]]]): List of rule violations
            **kwargs: Additional context for improvement

        Returns:
            str: The improved output
        """
        raise NotImplementedError("Subclasses must implement improve()")
