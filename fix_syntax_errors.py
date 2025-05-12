#!/usr/bin/env python3
"""
Script to fix actual syntax errors in the Sifaka codebase.

This script first identifies files with actual syntax errors,
then applies targeted fixes only to those files.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

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

def find_files_with_syntax_errors(directory: str) -> Dict[str, str]:
    """
    Find all Python files with syntax errors in a directory.

    Args:
        directory: Directory to search

    Returns:
        Dictionary mapping file paths to error messages
    """
    files_with_errors = {}
    python_files = find_python_files(directory)
    
    for file_path in python_files:
        error = check_syntax_errors(file_path)
        if error:
            files_with_errors[file_path] = error
    
    return files_with_errors

def fix_duplicate_def(content: str) -> str:
    """Fix duplicate def keyword."""
    return re.sub(r"def\s+def\s+", "def ", content)

def fix_mismatched_parentheses(content: str) -> str:
    """
    Fix mismatched parentheses in a very conservative way.
    Only fixes clear cases where a closing brace doesn't match an opening parenthesis.
    """
    # Fix closing brace } that doesn't match opening parenthesis (
    content = re.sub(r"\(([^{}]*)\}", r"(\1)", content)
    
    # Fix closing bracket ] that doesn't match opening parenthesis (
    content = re.sub(r"\(([^[\]]*)\]", r"(\1)", content)
    
    return content

def fix_file(file_path: str, error_message: str, dry_run: bool = False) -> Tuple[bool, List[str]]:
    """
    Fix syntax errors in a file based on the error message.

    Args:
        file_path: Path to the file to fix
        error_message: The syntax error message
        dry_run: If True, don't actually modify the file

    Returns:
        Tuple of (success, list of fixes applied)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False, []

    original_content = content
    fixes_applied = []

    # Apply fixes based on error message
    if "def def" in error_message or "def def" in content:
        new_content = fix_duplicate_def(content)
        if new_content != content:
            content = new_content
            fixes_applied.append("Fixed duplicate def keyword")
    
    if "does not match opening parenthesis" in error_message:
        new_content = fix_mismatched_parentheses(content)
        if new_content != content:
            content = new_content
            fixes_applied.append("Fixed mismatched parentheses")

    # Only write back if changes were made and not in dry run mode
    if content != original_content and not dry_run:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Verify that the fix worked
            error = check_syntax_errors(file_path)
            if error:
                print(f"Warning: File {file_path} still has syntax errors after fixing: {error}")
                return False, fixes_applied
            
            return True, fixes_applied
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            return False, fixes_applied

    return len(fixes_applied) > 0, fixes_applied

def main():
    """Main function to run the script."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Fix syntax errors in Python files")
    parser.add_argument("directory", nargs="?", default="sifaka", help="Directory to search for Python files")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify files")
    parser.add_argument("--file", help="Fix a specific file instead of a directory")
    args = parser.parse_args()

    if args.file:
        # Fix a specific file
        if not os.path.isfile(args.file):
            print(f"Error: {args.file} is not a file")
            sys.exit(1)
        
        error = check_syntax_errors(args.file)
        if not error:
            print(f"No syntax errors found in {args.file}")
            sys.exit(0)
        
        print(f"Fixing syntax error in {args.file}: {error}")
        success, fixes = fix_file(args.file, error, dry_run=args.dry_run)
        
        if success:
            print(f"Successfully fixed {args.file}")
            for fix in fixes:
                print(f"  - {fix}")
        else:
            print(f"Failed to fix {args.file}")
        
        sys.exit(0)

    # Fix all files in a directory
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        sys.exit(1)
    
    files_with_errors = find_files_with_syntax_errors(args.directory)
    print(f"Found {len(files_with_errors)} files with syntax errors")
    
    if not files_with_errors:
        print("No syntax errors found")
        sys.exit(0)
    
    # Fix each file with syntax errors
    fixed_files = 0
    for file_path, error in files_with_errors.items():
        print(f"Fixing {file_path}: {error}")
        success, fixes = fix_file(file_path, error, dry_run=args.dry_run)
        
        if success:
            fixed_files += 1
            print(f"  Successfully fixed")
            for fix in fixes:
                print(f"    - {fix}")
        else:
            print(f"  Failed to fix")
    
    print(f"\nSummary: Fixed {fixed_files} out of {len(files_with_errors)} files with syntax errors")
    if args.dry_run:
        print("Note: This was a dry run, no files were actually modified")

if __name__ == "__main__":
    main()
