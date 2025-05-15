#!/usr/bin/env python3
"""
Script to automatically fix time-related mypy issues in the codebase.

This script finds and replaces problematic patterns related to time operations
that cause mypy errors, such as conditional time expressions and string/float mismatches.
"""

import os
import re
import argparse
from typing import List, Pattern, Tuple

# Regular expression patterns for time-related issues
PATTERNS = [
    # time.time() if time else ""
    (
        r'time\.time\(\) if time else ""',
        r"time.time()",
    ),
    # time.time() if time else 0
    (
        r"time\.time\(\) if time else 0",
        r"time.time()",
    ),
    # (time.time() - start_time) if time else 0
    (
        r"\(time\.time\(\) - start_time\) if time else 0",
        r"(time.time() - start_time)",
    ),
    # start_time = time.time() if time else ""
    (
        r'start_time = time\.time\(\) if time else (?:""|0|0\.0)',
        r"start_time = time.time()",
    ),
    # processing_time_ms=time.time() if time else "" - start_time
    (
        r'processing_time_ms\s*=\s*time\.time\(\) if time else (?:""|0) - start_time',
        r"processing_time_ms=(time.time() - start_time) * 1000",
    ),
    # processing_time_ms=(time.time() - start_time) if time else 0
    (
        r"processing_time_ms\s*=\s*\(time\.time\(\) - start_time\) if time else 0",
        r"processing_time_ms=(time.time() - start_time) * 1000",
    ),
    # self._state_manager.set_metadata("creation_time", time.time() if time else "")
    (
        r'self\._state_manager\.set_metadata\("([^"]+)", time\.time\(\) if time else (?:""|0|0\.0)\)',
        r'self._state_manager.set_metadata("\1", time.time())',
    ),
    # self._state_manager.get_metadata("creation_time", time.time() if time else 0)
    (
        r'self\._state_manager\.get_metadata\("([^"]+)", time\.time\(\) if time else (?:""|0|0\.0)\)',
        r'self._state_manager.get_metadata("\1", time.time())',
    ),
    # Nested conditionals: if timeout and time.time() if time else "" - start_time > timeout:
    (
        r'if timeout and time\.time\(\) if time else "" - start_time > timeout:',
        r"if timeout and time.time() - start_time > timeout:",
    ),
    # Operator issues in time calculations
    (
        r'time\.time\(\) - start_time if time else (?:""|0)',
        r"(time.time() - start_time)",
    ),
    # processing_time_ms=time.time() - start_time if time else ""
    (
        r'processing_time_ms\s*=\s*time\.time\(\) - start_time if time else (?:""|0)',
        r"processing_time_ms=(time.time() - start_time) * 1000",
    ),
    # Process problematic list comprehensions with time checks
    (
        r'\[i\.lower\(\) if i else "" for i in ([^\]]+)\]',
        r"[i.lower() for i in \1 if i is not None]",
    ),
]


def fix_file(file_path: str, patterns: List[Tuple[Pattern, str]], dry_run: bool = False) -> bool:
    """
    Fix time-related issues in a file.

    Args:
        file_path: Path to the file to fix
        patterns: List of (pattern, replacement) tuples
        dry_run: If True, only print changes without writing to file

    Returns:
        True if any changes were made, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        print(f"Error reading file {file_path}: {e}")
        return False

    original_content = content
    for pattern_str, replacement in patterns:
        pattern = re.compile(pattern_str)
        content = pattern.sub(replacement, content)

    if content != original_content:
        if not dry_run:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Fixed {file_path}")
            except IOError as e:
                print(f"Error writing to file {file_path}: {e}")
                return False
        else:
            print(f"Would fix {file_path}")
        return True

    return False


def process_directory(
    directory: str,
    patterns: List[Tuple[Pattern, str]],
    extensions: List[str] = [".py"],
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Process all files in a directory recursively.

    Args:
        directory: Directory to process
        patterns: List of (pattern, replacement) tuples
        extensions: List of file extensions to process
        dry_run: If True, only print changes without writing to file

    Returns:
        (num_files_processed, num_files_changed)
    """
    compiled_patterns = [(re.compile(pattern), replacement) for pattern, replacement in patterns]
    num_files_processed = 0
    num_files_changed = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                num_files_processed += 1
                if fix_file(file_path, compiled_patterns, dry_run):
                    num_files_changed += 1

    return num_files_processed, num_files_changed


def main():
    parser = argparse.ArgumentParser(description="Fix time-related mypy issues in the codebase")
    parser.add_argument("directory", help="Directory to process")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print changes without writing to files"
    )
    parser.add_argument(
        "--extensions", default=".py", help="Comma-separated list of file extensions to process"
    )

    args = parser.parse_args()
    extensions = args.extensions.split(",")

    print(f"Processing {args.directory} {'(dry run)' if args.dry_run else ''}")
    num_files_processed, num_files_changed = process_directory(
        args.directory, PATTERNS, extensions, args.dry_run
    )

    print(f"Processed {num_files_processed} files, changed {num_files_changed} files")


if __name__ == "__main__":
    main()
