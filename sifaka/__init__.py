"""
Sifaka: A streamlined framework for LLM applications.

This module provides a clean, intuitive API for building reliable
LLM applications with validation and improvement capabilities.
"""

__version__ = "0.1.0"

# Initialize the registry if available
try:
    # The registry initialization is now in the main registry module
    from sifaka.registry import initialize_registry

    initialize_registry()
except ImportError:
    pass

# Import core components to make them available at the top level
from sifaka.chain import Chain

__all__ = ["Chain"]
