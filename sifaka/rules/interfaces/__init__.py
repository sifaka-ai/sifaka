"""
Rule interfaces for Sifaka.

This module defines the interfaces for rules and validators in the Sifaka framework.
These interfaces establish a common contract for rule behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Rule**: Base interface for all rules
   - **Validator**: Interface for validators
   - **RuleResultHandler**: Interface for rule result handlers
   - **RuleProtocol**: Simplified protocol for rules
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
