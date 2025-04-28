"""
Medical domain validation rules.

This module provides rules and validators for medical content validation,
including terminology validation, definition requirements, and term limits.
"""

from .config import MedicalConfig
from .validator import MedicalValidator

__all__ = [
    "MedicalConfig",
    "MedicalValidator",
]
