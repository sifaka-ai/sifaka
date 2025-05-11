#!/usr/bin/env python3
"""
Test script for the core/base.py module.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import BaseResult directly
from sifaka.core.base import BaseResult

# Create a BaseResult
result = BaseResult(
    passed=True,
    message="Test passed",
    metadata={"test": "core"},
)

# Print the result
print(f"BaseResult: {result}")

# Print success message
print('Successfully imported and tested BaseResult')
