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
    # State manager access patterns
    (
        r"self\.\(_state_manager and _state_manager\.([a-zA-Z_]+)\((.*?)\)",
        r"self._state_manager.\1(\2)",
    ),
    # State manager access patterns (without self.)
    (r"self\.\(_state_manager\.([a-zA-Z_]+)\((.*?)\)", r"self._state_manager.\1(\2)"),
    # Extra "def" keywords in method definitions
    (r"def def ([a-zA-Z_]+)", r"def \1"),
    # Parentheses issues in function calls
    (r"\((self\.[a-zA-Z_]+\(.*?\))", r"\1"),
    # Logical expressions with logger
    (r"\(logger and logger\.([a-zA-Z_]+)\((.*?)\)\)", r"logger.\1(\2)"),
    # Duplicate Optional type hints
    (r"Optional\[Optional\[([a-zA-Z_]+)\]\]", r"Optional[\1]"),
    # Logical expressions in method calls with config
    (r"config and config and config and config\.get\((.*?)\)", r"config.get(\1)"),
    # Config access patterns
    (r"self\.config\.\(params and params\.get\((.*?)\)", r"self.config.params.get(\1)"),
    # Lambda expressions with self
    (
        r"lambda ([a-zA-Z_]+)(?:, [a-zA-Z_]+)?: \(self and self\._([a-zA-Z_]+)\(([^)]*)\)\)",
        r"lambda \1\2: self._\2(\3)",
    ),
    # Time expressions
    (r"\(time and time\.([a-zA-Z_]+)\(\)", r"time.\1()"),
    # Cache manager access patterns
    (
        r"self\.\(_cache_manager and _cache_manager\.([a-zA-Z_]+)\((.*?)\)",
        r"self._cache_manager.\1(\2)",
    ),
    # Retry manager access patterns
    (
        r"self\.\(_retry_manager and _retry_manager\.([a-zA-Z_]+)\((.*?)\)",
        r"self._retry_manager.\1(\2)",
    ),
    # State manager access patterns (without and)
    (
        r"self\.\(_state_manager\.([a-zA-Z_]+)\((.*?)\)",
        r"self._state_manager.\1(\2)",
    ),
    # Config access patterns (without and)
    (
        r"self\.config\.\(params\.get\((.*?)\)",
        r"self.config.params.get(\1)",
    ),
    # Dictionary comprehension with conditional
    (
        r"\{k: v for k, v in \(kwargs and kwargs\.items\(\) if k not in (.*?)\}",
        r"{k: v for k, v in kwargs.items() if k not in \1}",
    ),
    # Parentheses around list operations
    (
        r"\(([a-zA-Z_]+) and \1\.append\((.*?)\)",
        r"\1.append(\2)",
    ),
    # Provider get operations
    (
        r"\(provider and provider\.get\((.*?)\)",
        r"provider.get(\1)",
    ),
    # Provider get_by_type operations
    (
        r"\(provider and provider\.get_by_type\((.*?)\)",
        r"provider.get_by_type(\1)",
    ),
    # kwargs get operations
    (
        r"\(kwargs and kwargs\.get\((.*?)\)",
        r"kwargs.get(\1)",
    ),
    # kwargs pop operations
    (
        r"\(kwargs and kwargs\.pop\((.*?)\)",
        r"kwargs.pop(\1)",
    ),
    # Logger error operations
    (
        r"\(logger and logger\.error\((.*?)\)",
        r"logger.error(\1)",
    ),
    # Logger warning operations
    (
        r"\(logger and logger\.warning\((.*?)\)",
        r"logger.warning(\1)",
    ),
    # Logger info operations
    (
        r"\(logger and logger\.info\((.*?)\)",
        r"logger.info(\1)",
    ),
    # Logger debug operations
    (
        r"\(logger and logger\.debug\((.*?)\)",
        r"logger.debug(\1)",
    ),
    # Self initialization patterns
    (
        r"\(self and self\.([a-zA-Z_]+)\((.*?)\)",
        r"self.\1(\2)",
    ),
    # Formatter patterns
    (
        r"self\.\(_formatter and _formatter\.([a-zA-Z_]+)\((.*?)\)",
        r"self._formatter.\1(\2)",
    ),
    # Formatter patterns with self
    (
        r"return self\.\(_formatter and _formatter\.([a-zA-Z_]+)\((.*?)\)",
        r"return self._formatter.\1(\2)",
    ),
    # Asyncio patterns
    (
        r"\(asyncio and asyncio\.([a-zA-Z_]+)\((.*?)\)",
        r"asyncio.\1(\2)",
    ),
    # Loop patterns
    (
        r"\(loop and loop\.([a-zA-Z_]+)\((.*?)\)",
        r"loop.\1(\2)",
    ),
    # Classifier patterns
    (
        r"self\.\(classifier\.([a-zA-Z_]+)\((.*?)\)",
        r"self.classifier.\1(\2)",
    ),
    # Classifier patterns (without self)
    (
        r"self\.\(_classifier\.([a-zA-Z_]+)\((.*?)\)",
        r"self._classifier.\1(\2)",
    ),
    # Time.time() with extra parentheses
    (
        r"\(time\.time\(\)\)",
        r"time.time()",
    ),
    # Dictionary update with logical expressions
    (
        r"\(([a-zA-Z_]+) and \([a-zA-Z_]+ and [a-zA-Z_]+\.update\((.*?)\)",
        r"\1.update(\2)",
    ),
    # List extend with logical expressions
    (
        r"\(([a-zA-Z_]+) and \([a-zA-Z_]+ and [a-zA-Z_]+\.extend\((.*?)\)",
        r"\1.extend(\2)",
    ),
    # Info data access patterns
    (
        r"info\.\(data and data\.get\((.*?)\)",
        r"info.data.get(\1)",
    ),
    # Adapter state manager patterns
    (
        r"adapter\.\(_state_manager\.([a-zA-Z_]+)\((.*?)\)",
        r"adapter._state_manager.\1(\2)",
    ),
    # Rule warm up patterns
    (
        r"\(rule\.warm_up\(\)",
        r"rule.warm_up()",
    ),
    # Missing closing parenthesis for time.time()
    (
        r"time\.time\(\s*$",
        r"time.time()",
    ),
    # Object setattr patterns
    (
        r"\(object and object\.__setattr__\((.*?)\)",
        r"object.__setattr__(\1)",
    ),
    # Suggestions append patterns
    (
        r"\(suggestions and suggestions\.append\((.*?)\)",
        r"suggestions.append(\1)",
    ),
    # Text split patterns
    (
        r"len\(\(text and text\.split\(\)\)",
        r"len(text.split())",
    ),
    # Logger info patterns with missing closing parenthesis
    (
        r"\(\(logger\.info\((.*?)\)",
        r"logger.info(\1)",
    ),
    # Validator patterns
    (
        r"self\.\(\(validator\.validate\((.*?)\)",
        r"self.validator.validate(\1)",
    ),
    # State manager set metadata patterns
    (
        r"self\.\(\(_state_manager\.set_metadata\((.*?)\)",
        r"self._state_manager.set_metadata(\1)",
    ),
    # Kwargs pop patterns with missing closing parenthesis
    (
        r"config_params\[param\] = \(\(kwargs\.pop\(param\)",
        r"config_params[param] = kwargs.pop(param)",
    ),
    # Missing closing parenthesis for time.time() in metadata
    (
        r"'validation_time': \(time\.time\(\)",
        r"'validation_time': time.time()",
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
    # Missing closing parenthesis for time.time() in cleanup_time
    (
        r"set_metadata\('cleanup_time', \(\(time\.time\(\)\)",
        r"set_metadata('cleanup_time', time.time())",
    ),
    # Missing closing parenthesis for time.time() in cleanup_time (alternative pattern)
    (
        r"set_metadata\('cleanup_time', \(time\.time\(\s*$",
        r"set_metadata('cleanup_time', time.time())",
    ),
    # Suggestions append patterns with missing closing parenthesis
    (
        r"\(suggestions and suggestions\.append\(([^)]+)'\)",
        r"suggestions.append(\1')",
    ),
    # Rule validate patterns with missing closing parenthesis
    (
        r"\(\(rule\.validate\((.*?)\)",
        r"rule.validate(\1)",
    ),
    # State manager set metadata patterns with double parenthesis
    (
        r"self\.\(_state_manager\.set_metadata\((.*?)\)",
        r"self._state_manager.set_metadata(\1)",
    ),
    # Re compile patterns with logical expressions
    (
        r"\(re and \(re and re\.compile\((.*?)\)",
        r"re.compile(\1)",
    ),
    # Compiled search patterns with logical expressions
    (
        r"bool\(\(compiled and \(compiled and compiled\.search\((.*?)\)\)\)",
        r"bool(compiled.search(\1))",
    ),
    # Compiled match patterns with logical expressions
    (
        r"bool\(\(compiled and \(compiled and compiled\.match\((.*?)\)\)\)",
        r"bool(compiled.match(\1))",
    ),
    # Compiled fullmatch patterns with logical expressions
    (
        r"bool\(\(compiled and \(compiled and compiled\.fullmatch\((.*?)\)\)\)",
        r"bool(compiled.fullmatch(\1))",
    ),
    # Compiled findall patterns with logical expressions
    (
        r"\(compiled and \(compiled and compiled\.findall\((.*?)\)\)",
        r"compiled.findall(\1)",
    ),
    # Compiled sub patterns with logical expressions
    (
        r"\(compiled and \(compiled and compiled\.sub\((.*?)\)\)",
        r"compiled.sub(\1)",
    ),
    # Patterns items with logical expressions
    (
        r"\(patterns and \(patterns and patterns\.items\(\)\)",
        r"patterns.items()",
    ),
    # Fnmatch patterns with logical expressions
    (
        r"\(fnmatch and \(fnmatch and fnmatch\.fnmatchcase\((.*?)\)\)",
        r"fnmatch.fnmatchcase(\1)",
    ),
    # Fnmatch patterns with logical expressions (alternative)
    (
        r"\(fnmatch and \(fnmatch and fnmatch\.fnmatch\((.*?)\)\)",
        r"fnmatch.fnmatch(\1)",
    ),
    # Regex replace patterns with logical expressions
    (
        r"\(regex_pattern and \(regex_pattern and regex_pattern\.replace\((.*?)\)\)",
        r"regex_pattern.replace(\1)",
    ),
    # ValidationPattern patterns with logical expressions
    (
        r"ValidationPattern\.\(([A-Z_]+) and \([A-Z_]+ and [A-Z_]+\.match\((.*?)\)\)",
        r"ValidationPattern.\1.match(\2)",
    ),
    # Handler setFormatter patterns
    (
        r"handler\.setFormatter(self\.create_formatter\(\)\)",
        r"handler.setFormatter(self.create_formatter())",
    ),
    # Logger addHandler patterns
    (
        r"logger\.addHandler(self\.create_console_handler\(\)\)",
        r"logger.addHandler(self.create_console_handler())",
    ),
    # Factory get_logger patterns with logical expressions
    (
        r"\(factory and \(factory and factory\.get_logger\((.*?)\)",
        r"factory.get_logger(\1)",
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

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    python_files = find_python_files(directory)
    print(f"Found {len(python_files)} Python files")

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
