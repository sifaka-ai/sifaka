from typing import Any, List
"""
Strategies for critics.

This package provides strategy implementations for critics in the Sifaka framework.
"""
from .improvement import ImprovementStrategy, DefaultImprovementStrategy
__all__: List[Any] = ['ImprovementStrategy', 'DefaultImprovementStrategy']
