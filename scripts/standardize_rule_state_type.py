#!/usr/bin/env python3
"""
Script to standardize state management type annotations in Sifaka rule implementations.

This script:
1. Finds all rule files that use _state: StateManager[RuleState] = PrivateAttr(default_factory=create_rule_state)
2. Changes it to _state_manager = PrivateAttr(default_factory=create_rule_state)
3. Updates all references to self._state.get_state() to self._state_manager.get_state()
4. Updates any other references to self._state to use self._state_manager instead

Usage:
    python standardize_rule_state_type.py <base_dir>
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Regular expressions for pattern matching
STATE_ATTR_PATTERN = r"_state\s*:\s*StateManager\[RuleState\]\s*=\s*PrivateAttr\(default_factory=create_rule_state\)"
STATE_ATTR_PATTERN_GENERIC = r"_state\s*:\s*StateManager\[\w+State\]\s*=\s*PrivateAttr\(default_factory="
STATE_GET_STATE_PATTERN = r"state\s*=\s*self\._state\.get_state\(\)"
STATE_REFERENCE_PATTERN = r"self\._state\b"
STATE_IS_INITIALIZED_PATTERN = r"self\._state\.is_initialized"
STATE_INITIALIZE_PATTERN = r"state\s*=\s*self\._state\.initialize\(\)"


def find_rule_files(base_dir: str) -> List[str]:
    """Find all Python files that might contain rule implementations with typed state."""
    rule_files = []

    # Focus on the sifaka/rules directory
    rules_dir = os.path.join(base_dir, "sifaka", "rules")
    if not os.path.exists(rules_dir):
        print(f"Rules directory not found at {rules_dir}")
        return []

    for root, _, files in os.walk(rules_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Check if the file might contain rule implementations with typed state
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Look for rule classes with typed state
                        if (
                            "class" in content
                            and "Rule" in content
                            and "_state: StateManager" in content
                        ):
                            rule_files.append(file_path)
                            print(f"Found rule file with typed state: {file_path}")
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
    return rule_files


def update_state_attr(file_path: str) -> bool:
    """
    Update _state: StateManager[RuleState] = PrivateAttr(default_factory=...) to
    _state_manager = PrivateAttr(default_factory=...).
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
            # Find all occurrences of _state: StateManager[XState] = PrivateAttr(default_factory=...)
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
                # Extract the state factory
                factory_match = re.search(r"default_factory=(\w+)", line)
                if factory_match:
                    factory = factory_match.group(1)
                    # Replace the line
                    new_line = f"_state_manager = PrivateAttr(default_factory={factory})"
                    # Replace the line in the content
                    new_content = content[:start_pos] + new_line + content[end_pos:]
                    # Update content for the next iteration
                    content = new_content

        # Write the updated content back to the file
        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated state attribute in {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error updating state attribute in {file_path}: {str(e)}")
        return False


def update_state_references(file_path: str) -> Tuple[int, int, int]:
    """
    Update references to self._state to use self._state_manager instead.
    
    Returns:
        Tuple of (get_state_count, is_initialized_count, initialize_count)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace self._state.get_state() with self._state_manager.get_state()
        new_content, get_state_count = re.subn(
            STATE_GET_STATE_PATTERN,
            "state = self._state_manager.get_state()",
            content
        )

        # Replace self._state.is_initialized with self._state_manager.is_initialized
        newer_content, is_initialized_count = re.subn(
            STATE_IS_INITIALIZED_PATTERN,
            "self._state_manager.is_initialized",
            new_content
        )

        # Replace state = self._state.initialize() with state = self._state_manager.get_state()
        newest_content, initialize_count = re.subn(
            STATE_INITIALIZE_PATTERN,
            "state = self._state_manager.get_state()",
            newer_content
        )

        # Write the updated content back to the file
        if newest_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(newest_content)
            print(f"Updated state references in {file_path}")
        return get_state_count, is_initialized_count, initialize_count
    except Exception as e:
        print(f"Error updating state references in {file_path}: {str(e)}")
        return 0, 0, 0


def process_file(file_path: str) -> None:
    """Process a file to standardize state management."""
    print(f"\nProcessing {file_path}")
    
    # Update state attribute
    attr_updated = update_state_attr(file_path)
    
    # Update references to self._state
    get_state_count, is_initialized_count, initialize_count = update_state_references(file_path)
    
    # Summary
    if attr_updated or get_state_count > 0 or is_initialized_count > 0 or initialize_count > 0:
        print(f"Summary for {file_path}:")
        if attr_updated:
            print("  - Updated state attribute to _state_manager")
        print(f"  - Updated {get_state_count} get_state() calls")
        print(f"  - Updated {is_initialized_count} is_initialized checks")
        print(f"  - Updated {initialize_count} initialize() calls")
    else:
        print(f"No changes needed for {file_path}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python standardize_rule_state_type.py <base_dir>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    if not os.path.isdir(base_dir):
        print(f"Directory {base_dir} does not exist")
        sys.exit(1)
    
    # Find rule files
    rule_files = find_rule_files(base_dir)
    print(f"Found {len(rule_files)} rule files with typed state")
    
    # Process each file
    for file_path in rule_files:
        process_file(file_path)
    
    print(f"\nProcessed {len(rule_files)} rule files")


if __name__ == "__main__":
    main()
