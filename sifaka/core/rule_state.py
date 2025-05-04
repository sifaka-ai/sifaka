"""
Rule state management utilities.

This module provides utility functions and classes for managing rule state
in the Sifaka framework.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel

from sifaka.core.classifier_state import ComponentState, StateManager


class RuleState(ComponentState):
    """
    State for rules.

    This class represents the state of a rule component.
    It includes common state variables used by rules.
    """

    validator: Optional[Any] = None
    cache: Dict[str, Any] = {}
    dependencies_loaded: bool = False


def create_rule_state(**kwargs: Any) -> StateManager:
    """
    Create a state manager for a rule.

    Args:
        **kwargs: Additional keyword arguments to pass to the state

    Returns:
        A state manager for a rule
    """
    return StateManager(
        initializer=lambda: RuleState(**kwargs),
        initialized=False,
        state=None,
    )
