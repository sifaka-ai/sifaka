"""
Registry Initialization Module

This module is responsible for initializing the component registry by importing
all component implementations, which triggers their factory function registration.
This ensures that all factory functions are registered before they are needed.

## Overview
The module imports all component implementation modules, triggering their registration
code during import. This allows the registry to be populated without explicit calls
to registration functions from client code.

## Circular Import Solution
This module is a key part of the circular import solution:

1. When the application starts, it calls initialize_registry()
2. This imports all implementation modules
3. During import, implementation modules register their factory functions
4. After initialization, factory modules can retrieve factory functions without imports
5. This breaks the circular dependency chain

The initialization happens once at startup, and subsequent imports of implementation
modules won't cause circular imports because their factory functions are already
registered.

## Usage
Import this module at application startup to ensure the registry is fully populated:

```python
# Import the initialization function
from sifaka.core import initialize_registry

# Initialize the registry
initialize_registry.initialize_registry()

# Now all factory functions are registered and can be used
from sifaka.critics.factories import create_critic
critic = create_critic("prompt", instructions="Evaluate this text")
```

## Environment Variables
- SIFAKA_SKIP_REGISTRY_INIT: Set to "true" to skip automatic initialization
- SIFAKA_SKIP_MODULES: Comma-separated list of module paths to skip during initialization
"""

import importlib
import logging
import os
from typing import Dict, List, Optional, Set
import sys

logger = logging.getLogger(__name__)

# List of modules that contain factory function registrations
REGISTRY_MODULES = [
    # Critics
    "sifaka.critics.implementations.prompt",
    "sifaka.critics.implementations.reflexion",
    "sifaka.critics.implementations.constitutional",
    "sifaka.critics.implementations.lac",
    # Rules
    "sifaka.rules.formatting.length",
    "sifaka.rules.content.prohibited",
    "sifaka.rules.content.safety",
    "sifaka.rules.content.sentiment",
    "sifaka.rules.formatting.structure",
    "sifaka.rules.formatting.format",
    # Classifiers
    "sifaka.classifiers.factories",
    "sifaka.classifiers.implementations.content.toxicity",
    "sifaka.classifiers.implementations.properties.language",
    # Model providers
    "sifaka.models.factories",
    # Retrievers
    "sifaka.retrieval.factories",
]

# Flag to track initialization state
_initialized = False


def initialize_registry(modules: Optional[List[str]] = None, force: bool = False) -> None:
    """
    Initialize the component registry by importing all component implementation modules.

    Args:
        modules: Optional list of module paths to import. If None, uses the default list.
        force: If True, will reinitialize the registry even if already initialized.
    """
    global _initialized

    # Skip if already initialized and not forced
    if _initialized and not force:
        logger.info("Registry already initialized, skipping")
        return

    # Use default modules if none provided
    if modules is None:
        # Check environment variable for skipping certain modules
        skip_modules_str = os.environ.get("SIFAKA_SKIP_MODULES", "")
        skip_modules = set(m.strip() for m in skip_modules_str.split(",") if m.strip())

        # Filter modules based on skip list
        modules = [m for m in REGISTRY_MODULES if m not in skip_modules]

    imported: Set[str] = set()
    failed: Dict[str, str] = {}

    for module_path in modules:
        if module_path in imported:
            continue

        try:
            # Try to import the module
            mod = importlib.import_module(module_path)
            imported.add(module_path)
            logger.debug(f"Successfully imported {module_path}")
        except ImportError as e:
            failed[module_path] = str(e)
            logger.warning(f"Failed to import {module_path}: {e}")
        except Exception as e:
            # Catch other exceptions during import to prevent complete failure
            failed[module_path] = f"Error during import: {str(e)}"
            logger.warning(f"Error importing {module_path}: {e}")

    # Log summary
    if failed:
        logger.warning(f"Failed to import {len(failed)} modules: {', '.join(failed.keys())}")

    logger.info(f"Initialized registry with {len(imported)} modules")
    _initialized = True


# Function to reset initialization state (for testing purposes)
def reset_initialization_state() -> None:
    """Reset the initialization state (mainly for testing)."""
    global _initialized
    _initialized = False


if __name__ == "__main__":
    # If this module is executed directly, initialize the registry
    logging.basicConfig(level=logging.INFO)
    initialize_registry()
