#!/usr/bin/env python3
"""
Script to fix common syntax errors in the Sifaka codebase.

This script fixes:
1. Missing closing parentheses in conditionals
2. Chained 'and' operators in conditionals (e.g., self.config and config and config...)
3. Import statements with 'and' operators
4. F-string errors with misplaced parentheses
5. Incorrect attribute access with parentheses (e.g., self.(_state_manager and _state_manager.get(...))
"""

import os
import re
import sys
from pathlib import Path

# Patterns to fix
PATTERNS = [
    # Fix missing closing parentheses in conditionals
    (
        r"if not isinstance\([^)]+, str\) or not \(([^)]+) and ([^)]+)(:)",
        r"if not isinstance(\1, str) or not (\2 and \3):",
    ),
    # Fix complex conditional expressions with missing parentheses
    (
        r'if not isinstance\(([^)]+), str\) or not ([^)]+)\.strip\(\) if ([^)]+) else ""\):',
        r'if not isinstance(\1, str) or not (\2.strip() if \3 else ""):',
    ),
    # Fix chained 'and' operators in conditionals for config.track_performance
    (
        r"(self\.config and config and config.+?)\.track_performance",
        r'hasattr(self, "config") and self.config.track_performance',
    ),
    # Fix import statements with 'and' operators
    (
        r"from sifaka\.utils\.config and config.+?\.critics import (.+)",
        r"from sifaka.utils.config.critics import \1",
    ),
    # Fix f-string errors with misplaced parentheses
    (
        r'f"([^"]+)\{str\(e\)\)\}"',
        r'f"\1{str(e)}"',
    ),
    # Fix incorrect attribute access with parentheses
    (
        r"self\.\(_state_manager and _state_manager\.([^)]+)\)",
        r"self._state_manager.\1",
    ),
    # Fix model_copy calls with 'and' operators
    (
        r"\(([A-Z_]+) and \1\.model_copy\(",
        r"\1.model_copy(",
    ),
    # Fix model_copy calls with parentheses
    (
        r"config = \(([A-Z_]+)\.model_copy\(\)",
        r"config = \1.model_copy()",
    ),
    # Fix updates.update(kwargs) with parentheses
    (
        r"\(updates\.update\(kwargs\)\)",
        r"updates.update(kwargs)",
    ),
    # Fix config.model_copy with 'and' operators
    (
        r"config and config and config and \(config\.model_copy\(",
        r"config.model_copy(",
    ),
    # Fix critic and critic.run pattern
    (
        r"\(([a-z_]+) and \1\.([a-z_]+)\(([^)]*)\)",
        r'\1.\2(\3) if \1 else ""',
    ),
    # Add missing Union import
    (
        r"from typing import Any, Dict, Optional\n",
        r"from typing import Any, Dict, Optional, Union\n",
    ),
    # Fix missing except clause in try blocks
    (
        r"try:\n(.*?)(?=\n\s*def|\n\s*class|\n\s*$)",
        r'try:\n\1\nexcept Exception as e:\n    raise ValueError(f"Failed to execute: {str(e)}") from e',
    ),
    # Fix metadata with mismatched parentheses
    (
        r'metadata=\{"error_type": type\(e\)\.__name__\),',
        r'metadata={"error_type": type(e).__name__},',
    ),
    # Fix issues with missing closing parentheses in if statements
    (
        r'if not self\._state_manager\.get\("initialized", False:',
        r'if not self._state_manager.get("initialized", False):',
    ),
    # Fix missing closing parentheses in get calls
    (
        r'cache = self\._state_manager\.get\("cache", \{\}\)',
        r'cache = self._state_manager.get("cache", {})',
    ),
    (
        r'cache = self\._state_manager\.get\("cache", \{',
        r'cache = self._state_manager.get("cache", {})',
    ),
    (
        r'cache = self\._state_manager\.get\("cache", \{\}\}\)',
        r'cache = self._state_manager.get("cache", {})',
    ),
    # Fix missing closing parentheses in update calls
    (
        r'self\._state_manager\.update\("([^"]+)", ([^)]+)',
        r'self._state_manager.update("\1", \2)',
    ),
    # Fix missing closing parentheses in get_metadata calls
    (
        r'([a-z_]+)_count = self\._state_manager\.get_metadata\("([a-z_]+)_count", ([0-9])',
        r'\1_count = self._state_manager.get_metadata("\2_count", \3)',
    ),
    # Fix missing closing parentheses in set_metadata calls
    (
        r'self\._state_manager\.set_metadata\("([a-z_]+)_count", ([a-z_]+)_count \+ 1',
        r'self._state_manager.set_metadata("\1_count", \2_count + 1)',
    ),
    # Fix missing closing parentheses in time.time() calls
    (
        r"time\.time\( - start_time\)",
        r"time.time() - start_time",
    ),
    # Fix missing closing parentheses in memory_manager calls
    (
        r'memory_manager = self\._state_manager\.get\("memory_manager"',
        r'memory_manager = self._state_manager.get("memory_manager")',
    ),
    # Fix missing closing parentheses in last_improvement_time
    (
        r'self\._state_manager\.set_metadata\("last_improvement_time", time\.time\(\) if time else ""',
        r'self._state_manager.set_metadata("last_improvement_time", time.time() if time else "")',
    ),
    # Fix missing closing parentheses in total_processing_time_ms
    (
        r'total_time = self\._state_manager\.get_metadata\("total_processing_time_ms", 0\.0',
        r'total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)',
    ),
    # Fix issues array with missing closing parentheses
    (
        r'issues=\[f"Processing error: \{str\(e\)\)"\],',
        r'issues=[f"Processing error: {str(e)}"],',
    ),
]


def fix_file(file_path):
    """Fix syntax errors in a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Apply all patterns
    for pattern, replacement in PATTERNS:
        content = re.sub(pattern, replacement, content)

    # Only write if changes were made
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed {file_path}")
        return True
    return False


def find_python_files(directory):
    """Find all Python files in a directory."""
    return list(Path(directory).glob("**/*.py"))


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python fix_syntax.py <directory or file>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        # Process directory
        python_files = find_python_files(path)
        fixed_count = 0

        for file_path in python_files:
            if fix_file(file_path):
                fixed_count += 1

        print(f"Fixed {fixed_count} files out of {len(python_files)}")
    elif os.path.isfile(path) and path.endswith(".py"):
        # Process single file
        if fix_file(path):
            print(f"Fixed {path}")
        else:
            print(f"No changes needed for {path}")
    else:
        print(f"Error: {path} is not a Python file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
