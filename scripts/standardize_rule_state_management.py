#!/usr/bin/env python3
"""
Script to standardize state management in Sifaka rule implementations.

This script:
1. Finds all rule files that use _state = PrivateAttr(default_factory=...)
2. Changes it to _state_manager = PrivateAttr(default_factory=create_rule_state)
3. Updates all references to self._state to use self._state_manager.get_state()
4. Updates any other references to self._state to use self._state_manager instead

Usage:
    python standardize_rule_state_management.py [--dry-run] [--verbose]
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Patterns for finding and replacing state management code
STATE_ATTR_PATTERN = r"_state\s*=\s*PrivateAttr\(default_factory=create_rule_state\)"
STATE_ATTR_PATTERN_GENERIC = r"_state\s*=\s*PrivateAttr\(default_factory=[^)]+\)"
STATE_ACCESS_PATTERN = r"self\._state\."
STATE_ACCESS_PATTERN_GET = r"self\._state\.get_state\(\)"
STATE_MANAGER_PATTERN = r"self\._state_manager\.get_state\(\)"


def find_rule_files(search_dirs: List[str]) -> List[str]:
    """
    Find all rule files that might need updating.
    
    Args:
        search_dirs: Directories to search for rule files
        
    Returns:
        List of file paths that might need updating
    """
    rule_files = []
    
    for search_dir in search_dirs:
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    # Check if the file might contain rule implementations
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            # Look for rule classes
                            if (
                                "class" in content
                                and "Rule" in content
                                and "_state = PrivateAttr" in content
                            ):
                                rule_files.append(file_path)
                                print(f"Found rule file with _state: {file_path}")
                    except Exception as e:
                        print(f"Error reading file {file_path}: {str(e)}")
    return rule_files


def update_state_attr(file_path: str, dry_run: bool = False) -> bool:
    """
    Update _state = PrivateAttr(default_factory=...) to
    _state_manager = PrivateAttr(default_factory=create_rule_state).
    
    Args:
        file_path: Path to the file to update
        dry_run: If True, don't actually update the file
        
    Returns:
        True if the file was updated, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Try the specific pattern first
        new_content = re.sub(
            STATE_ATTR_PATTERN,
            "_state_manager = PrivateAttr(default_factory=create_rule_state)",
            content,
        )

        # If no changes were made, try the generic pattern
        if new_content == content:
            # Find all occurrences of _state = PrivateAttr(default_factory=...)
            matches = re.finditer(STATE_ATTR_PATTERN_GENERIC, content)
            for match in matches:
                # Get the start position of the match
                start_pos = match.start()
                # Find the end of the line
                end_pos = content.find("\n", start_pos)
                if end_pos == -1:  # If no newline found, use the end of the string
                    end_pos = len(content)
                # Get the line
                line = content[start_pos:end_pos]
                # Replace _state with _state_manager
                new_line = line.replace("_state", "_state_manager")
                # Replace the line in the content
                new_content = content[:start_pos] + new_line + content[end_pos:]
                # Update content for the next iteration
                content = new_content

        # If the content was changed, write it back to the file
        if new_content != content:
            if not dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            return True
        
        return False
    except Exception as e:
        print(f"Error updating state attribute in {file_path}: {str(e)}")
        return False


def update_state_access(file_path: str, dry_run: bool = False) -> bool:
    """
    Update self._state to self._state_manager.get_state().
    
    Args:
        file_path: Path to the file to update
        dry_run: If True, don't actually update the file
        
    Returns:
        True if the file was updated, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace self._state.get_state() with self._state_manager.get_state()
        new_content = re.sub(
            STATE_ACCESS_PATTERN_GET,
            STATE_MANAGER_PATTERN,
            content,
        )

        # Replace other occurrences of self._state with self._state_manager.get_state()
        new_content = re.sub(
            STATE_ACCESS_PATTERN,
            "self._state_manager.get_state().",
            new_content,
        )

        # If the content was changed, write it back to the file
        if new_content != content:
            if not dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            return True
        
        return False
    except Exception as e:
        print(f"Error updating state access in {file_path}: {str(e)}")
        return False


def update_file(file_path: str, dry_run: bool = False, verbose: bool = False) -> Tuple[bool, bool]:
    """
    Update a file to use the standardized state management pattern.
    
    Args:
        file_path: Path to the file to update
        dry_run: If True, don't actually update the file
        verbose: If True, print more information
        
    Returns:
        Tuple of (attr_updated, access_updated)
    """
    if verbose:
        print(f"Updating {file_path}...")
    
    # Update state attribute
    attr_updated = update_state_attr(file_path, dry_run)
    if attr_updated:
        if verbose:
            print(f"  Updated state attribute in {file_path}")
    
    # Update state access
    access_updated = update_state_access(file_path, dry_run)
    if access_updated:
        if verbose:
            print(f"  Updated state access in {file_path}")
    
    return attr_updated, access_updated


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Standardize state management in Sifaka rules")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually update files")
    parser.add_argument("--verbose", action="store_true", help="Print more information")
    args = parser.parse_args()
    
    # Find rule files
    search_dirs = ["not_beam/sifaka/sifaka/rules"]
    rule_files = find_rule_files(search_dirs)
    
    # Update files
    updated_files = 0
    for file_path in rule_files:
        attr_updated, access_updated = update_file(file_path, args.dry_run, args.verbose)
        if attr_updated or access_updated:
            updated_files += 1
    
    # Print summary
    print(f"Found {len(rule_files)} rule files, updated {updated_files} files")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
