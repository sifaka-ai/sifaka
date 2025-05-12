#!/usr/bin/env python3
"""
Script to fix specific syntax error patterns in the Sifaka codebase.

This script targets the most common syntax errors:
1. self.(_var and _var.method()) -> self._var.method()
2. (var and var.method()) -> var.method()
3. def def method_name -> def method_name
4. Mismatched parentheses/braces/brackets
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Patterns to fix
PATTERNS = [
    # Pattern 1: self.(_var and _var.method()) -> self._var.method()
    (
        r"self\.\(([a-zA-Z_]+) and \1\.([a-zA-Z_]+)\(([^)]*)\)",
        r"self.\1.\2(\3)",
    ),
    
    # Pattern 2: (var and var.method()) -> var.method()
    (
        r"\(([a-zA-Z_\.]+) and \1\.([a-zA-Z_]+)\(([^)]*)\)",
        r"\1.\2(\3)",
    ),
    
    # Pattern 3: def def method_name -> def method_name
    (
        r"def def ([a-zA-Z_]+)",
        r"def \1",
    ),
    
    # Pattern 4: Fix closing brace } that doesn't match opening parenthesis (
    (
        r"\(([^{}]*)\}",
        r"(\1)",
    ),
    
    # Pattern 5: Fix closing bracket ] that doesn't match opening parenthesis (
    (
        r"\(([^[\]]*)\]",
        r"(\1)",
    ),
]

def fix_file(file_path: str, dry_run: bool = False) -> Tuple[int, List[str]]:
    """
    Fix syntax issues in a file.

    Args:
        file_path: Path to the file to fix
        dry_run: If True, don't actually modify the file

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

    # Only write back if changes were made and not in dry run mode
    if content != original_content and not dry_run:
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

def check_syntax_errors(file_path: str) -> Optional[str]:
    """
    Check if a file has syntax errors.

    Args:
        file_path: Path to the file to check

    Returns:
        Error message if there are syntax errors, None otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            compile(f.read(), file_path, "exec")
        return None
    except SyntaxError as e:
        return str(e)
    except Exception as e:
        return str(e)

def main():
    """Main function to run the script."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Fix syntax errors in Python files")
    parser.add_argument("directory", nargs="?", default="sifaka", help="Directory to search for Python files")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify files")
    parser.add_argument("--check", action="store_true", help="Only check for syntax errors, don't fix them")
    parser.add_argument("--file", help="Fix a specific file instead of a directory")
    args = parser.parse_args()

    # Get files to process
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: {args.file} is not a file")
            sys.exit(1)
        python_files = [args.file]
    else:
        if not os.path.isdir(args.directory):
            print(f"Error: {args.directory} is not a directory")
            sys.exit(1)
        python_files = find_python_files(args.directory)
    
    print(f"Found {len(python_files)} Python files")

    # Check for syntax errors if requested
    if args.check:
        errors = 0
        for file_path in python_files:
            error = check_syntax_errors(file_path)
            if error:
                print(f"Syntax error in {file_path}: {error}")
                errors += 1
        print(f"\nFound {errors} files with syntax errors")
        sys.exit(0)

    # Fix syntax errors
    total_fixes = 0
    files_fixed = 0

    for file_path in python_files:
        num_fixes, fixes = fix_file(file_path, dry_run=args.dry_run)
        if num_fixes > 0:
            files_fixed += 1
            total_fixes += num_fixes
            print(f"Fixed {num_fixes} issues in {file_path}")
            for fix in fixes:
                print(f"  - {fix}")

    print(f"\nSummary: Fixed {total_fixes} issues in {files_fixed} files")
    if args.dry_run:
        print("Note: This was a dry run, no files were actually modified")

if __name__ == "__main__":
    main()
