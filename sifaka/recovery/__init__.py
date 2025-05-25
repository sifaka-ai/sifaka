"""Recovery system for Sifaka chains.

This package provides intelligent recovery capabilities for chain execution failures,
including automatic checkpointing, recovery strategy analysis, and pattern-based
recovery suggestions.

Key components:
- RecoveryManager: Analyzes failures and suggests recovery strategies
- RecoveryStrategy: Enumeration of available recovery strategies
- RecoveryAction: Specific recovery actions with confidence scores

Example:
    ```python
    from sifaka.recovery import RecoveryManager, RecoveryStrategy
    from sifaka.storage.checkpoints import CachedCheckpointStorage
    
    # Create recovery manager
    storage = CachedCheckpointStorage(your_storage)
    recovery_manager = RecoveryManager(storage)
    
    # Analyze a failure and get recovery suggestions
    try:
        result = chain.run()
    except Exception as e:
        actions = recovery_manager.analyze_failure(last_checkpoint, e)
        best_action = actions[0]  # Highest confidence action
        print(f"Suggested recovery: {best_action.description}")
    ```
"""

from sifaka.recovery.manager import RecoveryManager, RecoveryStrategy, RecoveryAction

__all__ = [
    "RecoveryManager",
    "RecoveryStrategy", 
    "RecoveryAction"
]
