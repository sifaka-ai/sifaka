"""
Pytest configuration file for Sifaka tests.

This file contains pytest hooks and fixtures used across all test modules.
"""

import os
import sys
import pytest

from sifaka.utils.logging import get_logger, configure_logging

# Configure logging
configure_logging(level="INFO")
logger = get_logger(__name__)

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_configure(config):
    """
    Configure pytest before tests run.

    This is called once at the beginning of a test run.
    """
    # Apply compatibility patches for external libraries
    try:
        from sifaka.utils.patches import apply_all_patches

        apply_all_patches()
        logger.info("Successfully applied compatibility patches")
    except ImportError:
        logger.warning("Could not import patches module, skipping compatibility patches")
    except Exception as e:
        logger.warning(f"Failed to apply compatibility patches: {e}")
