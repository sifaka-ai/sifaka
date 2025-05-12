from typing import Any, List
"""
Managers for rules.

This package provides specialized managers for different aspects of rules:
- ValidationManager: Manages validation logic and rule execution
"""
from .validation import ValidationManager
__all__: List[Any] = ['ValidationManager']
