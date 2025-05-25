"""Chain module for Sifaka.

This module contains the modular chain architecture components:
- Chain: Main fluent API interface
- ChainConfig: Configuration management
- ChainOrchestrator: High-level workflow coordination
- ChainExecutor: Low-level execution logic
- RecoveryManager: Checkpointing and recovery
"""

# Import main Chain class for backward compatibility
from sifaka.core.chain.chain import Chain

__all__ = [
    "Chain",
]
