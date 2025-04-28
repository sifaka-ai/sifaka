"""
Legal domain validation rules.

This module provides rules and validators for legal content validation,
including citation validation, legal terminology requirements, and disclaimer checks.
"""

from .config import LegalConfig
from .validator import LegalValidator

__all__ = [
    "LegalConfig",
    "LegalValidator",
]
