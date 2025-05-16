"""
Configuration base classes for Sifaka components.

This module provides the base configuration classes used by components
throughout the Sifaka framework. It centralizes configuration handling
to ensure consistency and reduce code duplication.

This module is separate from base.py to avoid circular imports.
"""

from sifaka.core.base import BaseConfig as _BaseConfig

# Re-export BaseConfig to avoid circular imports
BaseConfig = _BaseConfig

__all__ = ["BaseConfig"]
