"""
Rule interfaces for Sifaka.

This module defines the interfaces for rules and validators in the Sifaka framework.
These interfaces establish a common contract for rule behavior, enabling better
modularity and extensibility.
"""

from .rule import RuleProtocol

__all__ = ["RuleProtocol"]
