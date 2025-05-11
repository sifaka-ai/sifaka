#!/usr/bin/env python3
"""
Test script for the utils/text.py module.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import text utilities directly
from sifaka.utils.text import (
    is_empty_text,
    handle_empty_text,
    handle_empty_text_for_classifier,
    ClassificationResult,
)

# Test is_empty_text
print(f"is_empty_text(''): {is_empty_text('')}")
print(f"is_empty_text(' '): {is_empty_text(' ')}")
print(f"is_empty_text('hello'): {is_empty_text('hello')}")

# Test handle_empty_text
result = handle_empty_text(
    text="",
    passed=False,
    message="Empty text",
    metadata={"error_type": "empty_text"},
)
print(f"handle_empty_text result: {result}")

# Test handle_empty_text_for_classifier
result = handle_empty_text_for_classifier(
    text="",
    label="unknown",
    confidence=0.0,
    metadata={"error_type": "empty_text"},
)
print(f"handle_empty_text_for_classifier result: {result}")

# Print success message
print('Successfully imported and tested text utilities')
