#!/usr/bin/env python3
"""
Script to automatically fix common syntax issues in the Sifaka codebase.

This script identifies and fixes the following patterns:
1. State manager access patterns: self.(_state_manager and _state_manager.get_state()) -> self._state_manager.get_state()
2. Parentheses issues in function calls: (self.method_name(param1, param2) -> self.method_name(param1, param2)
3. Logical expressions in method calls: config and config and config.get('key', default) -> config.get('key', default)
4. Missing closing parentheses in various contexts
5. Malformed dictionary definitions
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional

# Patterns to fix
PATTERNS = [
    # State manager access patterns
    (
        r"self\.\(_state_manager and _state_manager\.([a-zA-Z_]+)\((.*?)\)",
        r"self._state_manager.\1(\2)",
    ),
    # State manager access patterns (without self.)
    (
        r"self\.\(_state_manager\.([a-zA-Z_]+)\((.*?)\)",
        r"self._state_manager.\1(\2)",
    ),
    # Logger patterns
    (
        r"\(logger and logger\.([a-zA-Z_]+)\((.*?)\)",
        r"logger.\1(\2)",
    ),
    # JSON patterns
    (
        r"\(json and json\.([a-zA-Z_]+)\((.*?)\)",
        r"json.\1(\2)",
    ),
    # Loop patterns
    (
        r"\(loop and loop\.([a-zA-Z_]+)\((.*?)\)",
        r"loop.\1(\2)",
    ),
    # Adapter patterns
    (
        r"\(adapter and adapter\.([a-zA-Z_]+)\((.*?)\)",
        r"adapter.\1(\2)",
    ),
    # Validator patterns
    (
        r"state\.\(validator and validator\.([a-zA-Z_]+)\((.*?)\)",
        r"state.validator.\1(\2)",
    ),
    # Cache patterns
    (
        r"\(cache and cache\.([a-zA-Z_]+)\((.*?)\)",
        r"cache.\1(\2)",
    ),
    # Engine patterns
    (
        r"self\.\(_engine and _engine\.([a-zA-Z_]+)\((.*?)\)",
        r"self._engine.\1(\2)",
    ),
    # Dictionary get with logical expressions
    (
        r"\(([a-zA-Z_]+) and \1\.get\((.*?)\)",
        r"\1.get(\2)",
    ),
    # Dictionary items with logical expressions
    (
        r"\(([a-zA-Z_]+) and \1\.items\(\)",
        r"\1.items()",
    ),
    # Fix malformed dictionary definitions
    (
        r"return \{'([^']+)': ([^,]+), '([^']+)':\s*\n\s+([^,]+), '([^']+)':\s*\n\s+([^,]+), '([^']+)':\s*\n\s+([^,]+)",
        r"return {\n    '\1': \2,\n    '\3': \4,\n    '\5': \6,\n    '\7': \8",
    ),
    # Fix missing closing parenthesis for time.time()
    (
        r"time\.time\(\s*$",
        r"time.time()",
    ),
    # Fix missing closing parenthesis in various contexts
    (
        r"\(([a-zA-Z_]+)\.([a-zA-Z_]+)\((.*?)\)",
        r"\1.\2(\3)",
    ),
]


def fix_file(file_path: str) -> Tuple[int, List[str]]:
    """
    Fix syntax issues in a file.

    Args:
        file_path: Path to the file to fix

    Returns:
        Tuple of (number of fixes, list of fixed patterns)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content
    fixes_applied = []

    for pattern, replacement in PATTERNS:
        # Count occurrences before replacement
        matches = re.findall(pattern, content)
        if matches:
            # Apply the fix
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixes_applied.append(f"{pattern} -> {replacement}")

    # Only write back if changes were made
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    return len(fixes_applied), fixes_applied


def find_python_files(directory: str) -> List[str]:
    """
    Find all Python files in a directory recursively.

    Args:
        directory: Directory to search

    Returns:
        List of Python file paths
    """
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def main():
    """Main function to run the script."""
    # Default to 'sifaka' directory if no argument is provided
    directory = "sifaka"
    if len(sys.argv) > 1:
        directory = sys.argv[1]

    python_files = []
    if os.path.isdir(directory):
        python_files = find_python_files(directory)
        print(f"Found {len(python_files)} Python files")
    elif os.path.isfile(directory) and directory.endswith(".py"):
        python_files = [directory]
        print(f"Processing single file: {directory}")
    else:
        print(f"Error: {directory} is not a directory or Python file")
        sys.exit(1)

    total_fixes = 0
    files_fixed = 0

    for file_path in python_files:
        num_fixes, fixes = fix_file(file_path)
        if num_fixes > 0:
            files_fixed += 1
            total_fixes += num_fixes
            print(f"Fixed {num_fixes} issues in {file_path}")
            for fix in fixes:
                print(f"  - {fix}")

    print(f"\nSummary: Fixed {total_fixes} issues in {files_fixed} files")


if __name__ == "__main__":
    main()
