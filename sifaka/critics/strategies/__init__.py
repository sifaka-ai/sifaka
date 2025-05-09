"""
Strategies for critics.

This package provides strategy implementations for critics in the Sifaka framework.
"""

from .improvement import (
    ImprovementStrategy,
    DefaultImprovementStrategy,
)

__all__ = [
    "ImprovementStrategy",
    "DefaultImprovementStrategy",
]
