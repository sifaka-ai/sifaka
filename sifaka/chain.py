"""
Chain module for Sifaka.

This module provides components for orchestrating the validation and improvement
flow between models, rules, and critics.

This is a backward compatibility module that re-exports the components from
the chain package. New code should import directly from sifaka.chain.
"""

from .chain import (
    Chain,
    ChainResult,
    ChainExecutor,
    ChainOrchestrator,
    FeedbackFormatter,
    RetryManager,
)

__all__ = [
    "Chain",
    "ChainResult",
    "ChainExecutor",
    "ChainOrchestrator",
    "FeedbackFormatter",
    "RetryManager",
]
