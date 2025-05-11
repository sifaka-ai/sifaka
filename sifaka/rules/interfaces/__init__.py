"""
Rule interfaces for Sifaka.

This module provides interfaces for rules in the Sifaka framework.
These interfaces establish a common contract for rule behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Rule**: Base interface for all rules
   - **AsyncRule**: Interface for asynchronous rules
   - **RuleProtocol**: Simplified protocol for rules
   - **RuleResultHandler**: Interface for rule result handlers
   - **Validatable**: Interface for validatable objects

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from .rule import (
    Rule,
    AsyncRule,
    RuleProtocol,
    RuleResultHandler,
    Validatable,
)

__all__ = [
    "Rule",
    "AsyncRule",
    "RuleProtocol",
    "RuleResultHandler",
    "Validatable",
]
