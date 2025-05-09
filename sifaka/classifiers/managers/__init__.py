"""
Managers for classifiers.

This package is deprecated. Import state management components directly from sifaka.utils.state.

IMPORTANT: This package will be removed in a future version. All code should use sifaka.utils.state directly.
"""

import warnings

# Issue a deprecation warning
warnings.warn(
    "The sifaka.classifiers.managers package is deprecated. "
    "Use sifaka.utils.state directly instead. "
    "This package will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the correct StateManager
from sifaka.utils.state import StateManager  # noqa

__all__ = [
    "StateManager",
]
