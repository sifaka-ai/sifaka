"""
State management for classifiers.

This module is deprecated. Import StateManager directly from sifaka.utils.state instead.

IMPORTANT: This module will be removed in a future version. All code should use sifaka.utils.state directly.
"""

import warnings

# Issue a deprecation warning
warnings.warn(
    "The sifaka.classifiers.managers.state module is deprecated. "
    "Use sifaka.utils.state directly instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the correct StateManager
from sifaka.utils.state import StateManager  # noqa
