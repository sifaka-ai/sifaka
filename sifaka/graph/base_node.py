"""Base node class for Sifaka graph nodes.

This module provides a common base class for all Sifaka graph nodes
to resolve forward reference issues.
"""

from typing import Any

from pydantic_graph import BaseNode

from sifaka.core.thought import SifakaThought


class SifakaNode(BaseNode[SifakaThought, Any, SifakaThought]):
    """Base class for all Sifaka graph nodes.
    
    This class provides a common base for all nodes in the Sifaka workflow
    and helps resolve forward reference issues with PydanticAI graphs.
    """
    pass
