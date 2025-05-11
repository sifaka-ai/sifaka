"""
Test script to verify imports are working correctly.
"""

import sys
import os

# Add the parent directory to the path so we can import the modules directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting test...")

try:
    print("Importing directly from the module files...")

    # Import the modules directly
    from sifaka.core.managers.prompt_factories import (
        SelfRefineCriticPromptManager,
        ConstitutionalCriticPromptManager,
    )
    from sifaka.core.managers.prompt import PromptCriticPromptManager

    print("Import successful!")

except Exception as e:
    print(f"Import failed: {e}")
    import traceback

    traceback.print_exc()

print("Test completed.")
