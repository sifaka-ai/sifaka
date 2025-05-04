#!/usr/bin/env python3
"""
Script to standardize state management in Sifaka components.

This script:
1. Finds all files that use _state = PrivateAttr(default_factory=create_X_state)
2. Changes it to _state_manager = PrivateAttr(default_factory=create_X_state)
3. Updates all references to self._state.get_state() to self._state_manager.get_state()
4. Updates any other references to self._state to use self._state_manager instead

This script can be used to update:
- Rules (create_rule_state)
- Classifiers (create_classifier_state)
- Critics (create_critic_state)
- Models (create_model_state)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Regular expressions for pattern matching
STATE_GET_STATE_PATTERN = r"state\s*=\s*self\._state\.get_state\(\)"
STATE_REFERENCE_PATTERN = r"self\._state\b"

# Component types and their state factories
COMPONENT_TYPES = {
    "rule": {
        "dir": "rules",
        "state_factory": "create_rule_state",
        "class_pattern": "Rule",
    },
    "classifier": {
        "dir": "classifiers",
        "state_factory": "create_classifier_state",
        "class_pattern": "Classifier",
    },
    "critic": {
        "dir": "critics",
        "state_factory": "create_critic_state",
        "class_pattern": "Critic",
    },
    "model": {
        "dir": "models",
        "state_factory": "create_model_state",
        "class_pattern": "Provider",
    },
}


def find_component_files(base_dir: str, component_type: str) -> List[str]:
    """Find all Python files that might contain component implementations."""
    component_files = []
    
    # Get component-specific settings
    component_info = COMPONENT_TYPES.get(component_type)
    if not component_info:
        print(f"Unknown component type: {component_type}")
        return []
    
    component_dir = component_info["dir"]
    class_pattern = component_info["class_pattern"]
    
    # Focus on the specific component directory
    search_dir = os.path.join(base_dir, "sifaka", component_dir)
    if not os.path.exists(search_dir):
        print(f"{component_type.capitalize()} directory not found at {search_dir}")
        return []
    
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Check if the file might contain component implementations
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Look for component classes
                        if (
                            "class" in content
                            and class_pattern in content
                            and "_state = PrivateAttr" in content
                        ):
                            component_files.append(file_path)
                            print(f"Found {component_type} file with _state: {file_path}")
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
    return component_files


def update_state_attr(file_path: str, component_type: str) -> bool:
    """
    Update _state = PrivateAttr(default_factory=...) to 
    _state_manager = PrivateAttr(default_factory=...).
    """
    # Get component-specific settings
    component_info = COMPONENT_TYPES.get(component_type)
    if not component_info:
        print(f"Unknown component type: {component_type}")
        return False
    
    state_factory = component_info["state_factory"]
    state_attr_pattern = rf"_state\s*=\s*PrivateAttr\(default_factory={state_factory}\)"
    state_attr_pattern_generic = r"_state\s*=\s*PrivateAttr\(default_factory="
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Try the specific pattern first
        new_content = re.sub(
            state_attr_pattern,
            f"_state_manager = PrivateAttr(default_factory={state_factory})",
            content
        )
        
        # If no changes were made, try the generic pattern
        if new_content == content:
            # Find all occurrences of _state = PrivateAttr(default_factory=...)
            matches = re.finditer(state_attr_pattern_generic, content)
            for match in matches:
                # Get the start position of the match
                start_pos = match.start()
                # Find the end of the line
                end_pos = content.find("\n", start_pos)
                if end_pos == -1:  # If no newline found, use the end of the string
                    end_pos = len(content)
                
                # Extract the full line
                line = content[start_pos:end_pos]
                
                # Create the replacement line
                replacement = line.replace("_state", "_state_manager", 1)
                
                # Replace just this occurrence
                new_content = content[:start_pos] + replacement + content[start_pos + len(line):]
                content = new_content  # Update content for the next iteration
        
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


def update_state_references(file_path: str) -> Tuple[int, int]:
    """
    Update references to self._state to use self._state_manager instead.
    
    Returns:
        Tuple of (get_state_count, other_references_count)
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

        # Replace other references to self._state with self._state_manager
        newer_content, other_count = re.subn(
            STATE_REFERENCE_PATTERN,
            "self._state_manager",
            new_content
        )
        
        # Adjust the count to exclude the get_state replacements we already counted
        other_count -= get_state_count

        # Write the updated content back to the file
        if newer_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(newer_content)
            print(f"Updated state references in {file_path}: {get_state_count} get_state calls, {other_count} other references")
        return get_state_count, other_count
    except Exception as e:
        print(f"Error updating state references in {file_path}: {str(e)}")
        return 0, 0


def process_file(file_path: str, component_type: str) -> None:
    """Process a file to standardize state management."""
    print(f"\nProcessing {file_path}")
    
    # Update _state attribute
    attr_updated = update_state_attr(file_path, component_type)
    
    # Update references to self._state
    get_state_count, other_count = update_state_references(file_path)
    
    # Summary
    if attr_updated or get_state_count > 0 or other_count > 0:
        print(f"Summary for {file_path}:")
        if attr_updated:
            print("  - Updated _state attribute to _state_manager")
        print(f"  - Updated {get_state_count} get_state() calls")
        print(f"  - Updated {other_count} other references to self._state")
    else:
        print(f"No changes needed for {file_path}")


def main():
    """Main function."""
    if len(sys.argv) < 3:
        print("Usage: python standardize_state_management.py <base_dir> <component_type>")
        print("Component types: rule, classifier, critic, model, all")
        sys.exit(1)

    base_dir = sys.argv[1]
    component_type = sys.argv[2].lower()
    
    if not os.path.isdir(base_dir):
        print(f"Directory {base_dir} does not exist")
        sys.exit(1)
    
    # Process all component types or just the specified one
    component_types = list(COMPONENT_TYPES.keys()) if component_type == "all" else [component_type]
    
    for comp_type in component_types:
        if comp_type not in COMPONENT_TYPES:
            print(f"Unknown component type: {comp_type}")
            continue
            
        print(f"\n=== Processing {comp_type} components ===\n")
        
        # Find component files
        component_files = find_component_files(base_dir, comp_type)
        print(f"Found {len(component_files)} {comp_type} files with _state attribute")
        
        # Process each file
        for file_path in component_files:
            process_file(file_path, comp_type)


if __name__ == "__main__":
    main()
