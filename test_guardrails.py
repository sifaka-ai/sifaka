#!/usr/bin/env python
"""Test script to check if Guardrails is correctly installed and importable."""

print("Trying to import from Guardrails...")

try:
    import guardrails

    print(f"✅ Successfully imported guardrails module")

    try:
        from guardrails.hub import RegexMatch

        print(f"✅ Successfully imported RegexMatch from guardrails.hub")
    except ImportError as e:
        print(f"❌ Failed to import RegexMatch from guardrails.hub: {e}")

    try:
        from guardrails.validator_base import Validator

        print(f"✅ Successfully imported Validator from guardrails.validator_base")
    except ImportError as e:
        print(f"❌ Failed to import Validator from guardrails.validator_base: {e}")

except ImportError as e:
    print(f"❌ Failed to import guardrails: {e}")

print("\nPython path:")
import sys

for path in sys.path:
    print(f"  - {path}")
