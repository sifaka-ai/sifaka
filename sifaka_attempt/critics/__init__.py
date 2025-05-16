"""
Critics for the Sifaka library.

This package provides critic classes that evaluate and improve text.
"""

from typing import Protocol, List, Optional, runtime_checkable


@runtime_checkable
class CriticProtocol(Protocol):
    """
    Protocol for critics that evaluate and improve text.

    Critics implement this protocol to evaluate text and provide improvements
    based on issues found.
    """

    def critique(self, text: str) -> dict:
        """
        Evaluate text and provide feedback.

        Args:
            text: The text to evaluate

        Returns:
            A dictionary with feedback, including a score, issues, and suggestions
        """
        ...

    def improve(self, text: str, issues: Optional[List[str]] = None) -> str:
        """
        Improve text based on issues found.

        Args:
            text: The text to improve
            issues: Optional list of issues to address

        Returns:
            Improved text
        """
        ...


# Import critic implementations
from .prompt import PromptCritic
from .reflexion import ReflexionCritic
from .constitutional import ConstitutionalCritic
from .self_rag import SelfRAGCritic
from .self_refine import SelfRefineCritic
from .lac import LACCritic

# Export critics
__all__ = [
    "CriticProtocol",
    "PromptCritic",
    "ReflexionCritic",
    "ConstitutionalCritic",
    "SelfRAGCritic",
    "SelfRefineCritic",
    "LACCritic",
]
