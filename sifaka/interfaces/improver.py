"""
Improver Protocol Module

This module defines the protocol for improvers in the Sifaka framework.
Improvers are components that improve text based on feedback or criteria.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .component import ComponentProtocol


@runtime_checkable
class ImproverProtocol(ComponentProtocol, Protocol):
    """
    Protocol for improvers in Sifaka.

    Improvers are components that improve text based on feedback or criteria.
    They take text as input and return improved versions of the text.
    """

    def improve(self, text: str, feedback: Optional[str] = None, **kwargs: Any) -> Any:
        """
        Improve the given text based on feedback or criteria.

        Args:
            text: The text to improve
            feedback: Optional feedback to guide improvement
            **kwargs: Additional arguments for improvement

        Returns:
            Improved text or improvement results
        """
        ...
