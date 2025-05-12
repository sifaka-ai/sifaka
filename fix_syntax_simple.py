#!/usr/bin/env python3
"""
Script to automatically fix common syntax issues in the Sifaka codebase.

This script identifies and fixes the following patterns:
1. State manager access patterns: self.(_state_manager and _state_manager.get_state()) -> self._state_manager.get_state()
2. Extra "def" keywords in method definitions: def def method_name -> def method_name
3. Parentheses issues in function calls: (self.method_name(param1, param2) -> self.method_name(param1, param2)
4. Duplicate Optional type hints: Optional[Optional[str]] -> Optional[str]
5. Logical expressions in method calls: config and config and config.get('key', default) -> config.get('key', default)
6. Config access patterns: self.config.(params and params.get('key', default) -> self.config.params.get('key', default)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional

# Patterns to fix
PATTERNS = [
    # Fix closing parenthesis } that doesn't match opening parenthesis (
    (
        r"\(([^{}]*)\}",
        r"(\1)",
    ),
    # Fix closing parenthesis ] that doesn't match opening parenthesis (
    (
        r"\(([^[\]]*)\]",
        r"(\1)",
    ),
    # Fix closing parenthesis ) that doesn't match opening parenthesis {
    (
        r"\{([^{}]*)\)",
        r"{\1}",
    ),
    # Fix closing parenthesis ) that doesn't match opening parenthesis [
    (
        r"\[([^[\]]*)\)",
        r"[\1]",
    ),
    # Fix unmatched closing parenthesis )
    (
        r"([^(]*)\)([^)]*)$",
        r"\1\2",
    ),
    # Fix unclosed opening parenthesis (
    (
        r"\(([^)]*?)$",
        r"(\1)",
    ),
    # Fix f-string unmatched )
    (
        r"f\"([^\"]*)\)([^\"]*)\"",
        r"f\"\1\2\"",
    ),
    # Fix common state manager access patterns
    (
        r"self\._state and _state\.get\(([^)]+)\)",
        r"self._state.get(\1)",
    ),
    # Fix common config access patterns
    (
        r"self\.config and config and config\.get\(([^)]+)\)",
        r"self.config.get(\1)",
    ),
    # Fix common logical expressions in method calls
    (
        r"config and config\.get\(([^)]+)\)",
        r"config.get(\1)",
    ),
    # Fix common logical expressions with logger
    (
        r"logger and logger\.(debug|info|warning|error)\(([^)]+)\)",
        r"logger.\1(\2)",
    ),
    # Fix common patterns with missing closing parenthesis
    (
        r"time\.time\(\s*$",
        r"time.time()",
    ),
    # Fix common patterns with extra parentheses
    (
        r"\(\(([a-zA-Z_]+\.[a-zA-Z_]+\([^)]*\))\)",
        r"\1",
    ),
    # Fix state manager patterns
    (
        r"self\.\(_state_manager and _state_manager\.([a-zA-Z_]+)\(([^)]*)\)",
        r"self._state_manager.\1(\2)",
    ),
    # Fix config access patterns
    (
        r"self\.config\.\(params and params\.get\(([^)]*)\)",
        r"self.config.params.get(\1)",
    ),
    # Fix bias.py specific issue
    (
        r"self\.config and config and config and config and config and config and config and config and config and config and config and config and config and config and config and config\.\(params and params\.get\(([^)]*)\)",
        r"self.config.params.get(\1)",
    ),
    # Re compile patterns
    (
        r"\(re and re\.compile\(([^)]+)\)",
        r"re.compile(\1)",
    ),
    # Compiled search patterns
    (
        r"\(compiled and compiled\.search\(([^)]+)\)",
        r"compiled.search(\1)",
    ),
    # Compiled match patterns
    (
        r"\(compiled and compiled\.match\(([^)]+)\)",
        r"compiled.match(\1)",
    ),
    # Compiled fullmatch patterns
    (
        r"\(compiled and compiled\.fullmatch\(([^)]+)\)",
        r"compiled.fullmatch(\1)",
    ),
    # Compiled findall patterns
    (
        r"\(compiled and compiled\.findall\(([^)]+)\)",
        r"compiled.findall(\1)",
    ),
    # Compiled sub patterns
    (
        r"\(compiled and compiled\.sub\(([^)]+)\)",
        r"compiled.sub(\1)",
    ),
    # Patterns items
    (
        r"\(patterns and patterns\.items\(\)",
        r"patterns.items()",
    ),
    # Fnmatch patterns
    (
        r"\(fnmatch and fnmatch\.fnmatchcase\(([^)]+)\)",
        r"fnmatch.fnmatchcase(\1)",
    ),
    # Fnmatch patterns (alternative)
    (
        r"\(fnmatch and fnmatch\.fnmatch\(([^)]+)\)",
        r"fnmatch.fnmatch(\1)",
    ),
    # Regex replace patterns
    (
        r"\(regex_pattern and regex_pattern\.replace\(([^)]+)\)",
        r"regex_pattern.replace(\1)",
    ),
    # ValidationPattern patterns
    (
        r"ValidationPattern\.\(([A-Z_]+) and ([A-Z_]+)\.match\(([^)]+)\)",
        r"ValidationPattern.\1.match(\3)",
    ),
    # Handler setFormatter patterns
    (
        r"handler\.setFormatter(self\.create_formatter\(\))",
        r"handler.setFormatter(self.create_formatter())",
    ),
    # Logger addHandler patterns
    (
        r"logger\.addHandler(self\.create_console_handler\(\))",
        r"logger.addHandler(self.create_console_handler())",
    ),
    # Factory get_logger patterns
    (
        r"\(factory and factory\.get_logger\(([^)]+)\)",
        r"factory.get_logger(\1)",
    ),
    # Missing closing parenthesis for time.time() in cleanup_time
    (
        r"set_metadata\('cleanup_time', \(time\.time\(\)",
        r"set_metadata('cleanup_time', time.time())",
    ),
    # State manager set metadata patterns
    (
        r"self\.\(_state_manager\.set_metadata\(([^)]+)\)",
        r"self._state_manager.set_metadata(\1)",
    ),
    # Missing closing parenthesis for time.time() in set_metadata
    (
        r"set_metadata\('creation_time', time\.time\(\s*$",
        r"set_metadata('creation_time', time.time())",
    ),
    # Missing closing parenthesis for time.time() in start_time
    (
        r"start_time = \(time\.time\(\s*$",
        r"start_time = time.time()",
    ),
    # Suggestions append patterns
    (
        r"\(suggestions and suggestions\.append\(([^)]+)\)",
        r"suggestions.append(\1)",
    ),
    # Object setattr patterns
    (
        r"\(object and object\.__setattr__\(([^)]+)\)",
        r"object.__setattr__(\1)",
    ),
    # Text split patterns
    (
        r"len\(\(text and text\.split\(\)\)",
        r"len(text.split())",
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
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Error reading {file_path}: UnicodeDecodeError")
        return 0, []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, []

    original_content = content
    fixes_applied = []

    for pattern, replacement in PATTERNS:
        try:
            # Apply the fix
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                fixes_applied.append(f"{pattern} -> {replacement}")
        except re.error as e:
            print(f"Error applying pattern {pattern} to {file_path}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error applying pattern {pattern} to {file_path}: {e}")
            continue

    # Only write back if changes were made
    if content != original_content:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            return 0, []

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
