#!/usr/bin/env python
"""
Script to fix the '_initialize_state undefined in superclass' error
by modifying relevant files to check if super() has the method before calling it.
"""

import os
import sys
from typing import List, Tuple, Optional

# Files to fix - these all have the _initialize_state error
FILES_TO_FIX = [
    "sifaka/adapters/chain/formatter.py",
    "sifaka/adapters/chain/improver.py",
    "sifaka/adapters/chain/model.py",
    "sifaka/adapters/chain/validator.py",
    "sifaka/chain/engine.py",
    "sifaka/chain/managers/cache.py",
    "sifaka/chain/managers/retry.py",
    "sifaka/classifiers/engine.py",
    "sifaka/rules/managers/validation.py",
    "sifaka/rules/validators.py",
]


def find_initialize_state_method(
    lines: List[str],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Find the _initialize_state method in the file and the line with super() call.

    Args:
        lines: Lines of the file

    Returns:
        Tuple of (method_start_line, super_call_line, indentation_level)
    """
    method_start = None
    super_call_line = None
    indent_level = None

    for i, line in enumerate(lines):
        if "def _initialize_state" in line:
            method_start = i
            # Extract indentation
            indent_level = len(line) - len(line.lstrip())

        if (
            method_start is not None
            and "super()._initialize_state()" in line
            and super_call_line is None
        ):
            super_call_line = i

        # If we've found both, we can stop
        if method_start is not None and super_call_line is not None:
            break

        # If we've moved on to another method, reset
        if method_start is not None and "def " in line and i > method_start + 2:
            break

    return method_start, super_call_line, indent_level


def fix_file(filepath: str) -> bool:
    """
    Fix the file by replacing direct super()._initialize_state() calls with a check.

    Args:
        filepath: Path to the file to fix

    Returns:
        True if the file was modified, False otherwise
    """
    try:
        # Read file
        with open(filepath, "r") as f:
            lines = f.readlines()

        # Check if the file already has the fix
        if any('hasattr(super(), "_initialize_state")' in line for line in lines):
            print(f"File already fixed: {filepath}")
            return False

        # Find the method and super call
        method_start, super_call_line, indent_level = find_initialize_state_method(lines)

        if method_start is None or super_call_line is None or indent_level is None:
            print(f"Could not locate _initialize_state method or super call in: {filepath}")
            return False

        # Create indentation string
        indent = " " * indent_level

        # Replace the super call with the check
        original_line = lines[super_call_line]
        check_line = f"{indent}# Check if super has _initialize_state method before calling it\n"
        if_line = f'{indent}if hasattr(super(), "_initialize_state"):\n'
        super_line = f"{indent}    super()._initialize_state()\n"

        # Replace the line
        lines[super_call_line] = check_line + if_line + super_line

        # Write the modified content back
        with open(filepath, "w") as f:
            f.writelines(lines)

        print(f"Fixed: {filepath}")
        return True

    except Exception as e:
        print(f"Error fixing {filepath}: {str(e)}")
        return False


def main() -> None:
    """Fix all files with _initialize_state issue."""
    # Change to the project root directory if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(script_dir, FILES_TO_FIX[0])):
        # We're not in the project root, try to find it
        if os.path.exists("sifaka"):
            # We're already in the project root
            pass
        else:
            print("Script must be run from the project root directory")
            sys.exit(1)

    # Fix all files
    fixed_count = 0
    for filepath in FILES_TO_FIX:
        if fix_file(filepath):
            fixed_count += 1

    print(f"Fixed {fixed_count} files of {len(FILES_TO_FIX)}")


if __name__ == "__main__":
    main()
