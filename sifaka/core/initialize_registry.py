"""
Registry Initialization Module

This module is responsible for initializing the component registry by importing
all component implementations, which triggers their factory function registration.
This ensures that all factory functions are registered before they are needed.

## Overview
The module imports all component implementation modules, triggering their registration
code during import. This allows the registry to be populated without explicit calls
to registration functions from client code.

## Usage
Import this module at application startup to ensure the registry is fully populated:

```python
from sifaka.core.initialize_registry import initialize_registry

# Initialize the registry
initialize_registry()

# Now all factory functions are registered and can be used
from sifaka.core.factories import create_critic
critic = create_critic("prompt")
```
"""

import importlib
import logging
from typing import Dict, List, Optional, Set
import sys

logger = logging.getLogger(__name__)

# List of modules that contain factory function registrations
REGISTRY_MODULES = [
    # Critics
    "sifaka.critics.implementations.prompt",
    "sifaka.critics.implementations.reflexion",
    "sifaka.critics.implementations.constitutional",
    "sifaka.critics.implementations.self_refine",
    "sifaka.critics.implementations.self_rag",
    "sifaka.critics.implementations.lac",
    # Rules
    "sifaka.rules.formatting.length",
    "sifaka.rules.content.prohibited",
    "sifaka.rules.content.safety",
    "sifaka.rules.content.sentiment",
    "sifaka.rules.formatting.structure",
    "sifaka.rules.formatting.format",
    "sifaka.rules.formatting.format.markdown",
    "sifaka.rules.formatting.format.json",
    "sifaka.rules.formatting.format.plain_text",
    # Classifiers
    "sifaka.classifiers.factories",
    "sifaka.classifiers.implementations.factories",
    "sifaka.classifiers.implementations.content.toxicity",
    "sifaka.classifiers.implementations.properties.language",
    # Model providers
    "sifaka.models.factories",
    # Retrievers
    "sifaka.retrieval.factories",
]

_initialized = False


def initialize_registry(modules: Optional[List[str]] = None) -> None:
    """
    Initialize the component registry by importing all component implementation modules.

    Args:
        modules: Optional list of module paths to import. If None, uses the default list.
    """
    global _initialized

    if _initialized:
        logger.info("Registry already initialized, skipping")
        return

    if modules is None:
        modules = REGISTRY_MODULES

    imported: Set[str] = set()
    failed: Dict[str, str] = {}

    for module_path in modules:
        if module_path in imported:
            continue

        try:
            importlib.import_module(module_path)
            imported.add(module_path)
            logger.debug(f"Successfully imported {module_path}")
        except ImportError as e:
            failed[module_path] = str(e)
            logger.warning(f"Failed to import {module_path}: {e}")

    # Log summary
    if failed:
        logger.warning(f"Failed to import {len(failed)} modules: {', '.join(failed.keys())}")

    logger.info(f"Initialized registry with {len(imported)} modules")
    _initialized = True


if __name__ == "__main__":
    # If this module is executed directly, initialize the registry
    logging.basicConfig(level=logging.INFO)
    initialize_registry()
