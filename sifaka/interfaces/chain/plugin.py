"""
Plugin interfaces for chain components.

This module defines the ChainPlugin interface for plugins in the chain system.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ChainPlugin(Protocol):
    """
    Interface for plugins in the chain system.
    """

    def initialize(self) -> None:
        """
        Initialize the plugin.
        """
        ...

    @property
    def name(self) -> str:
        """
        Get the plugin name.

        Returns:
            The name of the plugin
        """
        ...
