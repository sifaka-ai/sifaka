#!/usr/bin/env python3
"""
Script to standardize direct state initialization in Sifaka components.

This script:
1. Finds all files that use direct state initialization like self._state = ClassifierState()
2. Changes it to use the StateManager pattern with _state_manager
3. Updates all references to self._state to use self._state_manager.get_state()

This script can be used to update:
- Classifiers (ClassifierState)
- Critics (CriticState)
- Rules (RuleState)
- Models (ModelState)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Regular expressions for pattern matching
DIRECT_STATE_INIT_PATTERN = r"self\._state\s*=\s*(\w+State)\(\)"
STATE_REFERENCE_PATTERN = r"self\._state\b"

# Component types and their state factories
COMPONENT_TYPES = {
    "classifier": {
        "dir": "classifiers",
        "state_class": "ClassifierState",
        "state_factory": "create_classifier_state",
    },
    "critic": {
        "dir": "critics",
        "state_class": "CriticState",
        "state_factory": "create_critic_state",
    },
    "rule": {
        "dir": "rules",
        "state_class": "RuleState",
        "state_factory": "create_rule_state",
    },
    "model": {
        "dir": "models",
        "state_class": "ModelState",
        "state_factory": "create_model_state",
    },
}


def find_component_files(base_dir: str, component_type: str) -> List[str]:
    """Find all Python files that might contain direct state initialization."""
    component_files = []
    
    # Get component-specific settings
    component_info = COMPONENT_TYPES.get(component_type)
    if not component_info:
        print(f"Unknown component type: {component_type}")
        return []
    
    component_dir = component_info["dir"]
    state_class = component_info["state_class"]
    
    # Focus on the specific component directory
    search_dir = os.path.join(base_dir, "sifaka", component_dir)
    if not os.path.exists(search_dir):
        print(f"{component_type.capitalize()} directory not found at {search_dir}")
        return []
    
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Check if the file might contain direct state initialization
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Look for direct state initialization
                        if "self._state = " in content and state_class in content:
                            component_files.append(file_path)
                            print(f"Found {component_type} file with direct state init: {file_path}")
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
    return component_files


def update_direct_state_init(file_path: str, component_type: str) -> bool:
    """
    Update direct state initialization to use StateManager.
    
    Args:
        file_path: Path to the file to update
        component_type: Type of component (classifier, critic, rule, model)
        
    Returns:
        True if the file was updated, False otherwise
    """
    # Get component-specific settings
    component_info = COMPONENT_TYPES.get(component_type)
    if not component_info:
        print(f"Unknown component type: {component_type}")
        return False
    
    state_factory = component_info["state_factory"]
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Add import for state factory if needed
        if state_factory not in content:
            import_pattern = r"from sifaka\.utils\.state import (\w+)"
            match = re.search(import_pattern, content)
            if match:
                # Add state factory to existing import
                new_import = f"from sifaka.utils.state import {match.group(1)}, {state_factory}"
                content = re.sub(import_pattern, new_import, content)
            else:
                # Add new import
                import_line = f"from sifaka.utils.state import {state_factory}\n"
                # Add after other imports
                last_import = re.search(r"^import.*$|^from.*$", content, re.MULTILINE)
                if last_import:
                    pos = last_import.end()
                    content = content[:pos] + "\n" + import_line + content[pos:]
                else:
                    # Add at the beginning
                    content = import_line + content
        
        # Add _state_manager attribute
        class_pattern = r"class\s+(\w+).*:"
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            class_start = match.end()
            
            # Check if this class has direct state initialization
            init_pattern = rf"def\s+__init__.*?self\._state\s*=\s*\w+State\(\)"
            if re.search(init_pattern, content[class_start:], re.DOTALL):
                # Add _state_manager attribute
                attr_line = f"\n    # State management using StateManager\n    _state_manager = PrivateAttr(default_factory={state_factory})\n"
                
                # Find where to insert the attribute
                class_body_start = content.find(":", class_start) + 1
                next_line_start = content.find("\n", class_body_start) + 1
                
                # Insert the attribute
                content = content[:next_line_start] + attr_line + content[next_line_start:]
        
        # Update direct state initialization in __init__ methods
        init_pattern = r"(def\s+__init__.*?)(self\._state\s*=\s*\w+State\(\))(.*?)(self\._state\.initialized\s*=\s*False)"
        content = re.sub(
            init_pattern,
            r"\1# State is managed by StateManager, no need to initialize here\3# Initialization is handled by StateManager",
            content,
            flags=re.DOTALL
        )
        
        # Write the updated content back to the file
        if content != open(file_path, "r", encoding="utf-8").read():
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated direct state initialization in {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error updating direct state initialization in {file_path}: {str(e)}")
        return False


def update_state_references(file_path: str) -> int:
    """
    Update references to self._state to use self._state_manager.get_state().
    
    Args:
        file_path: Path to the file to update
        
    Returns:
        Number of references updated
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Replace self._state with state = self._state_manager.get_state() at the beginning of methods
        method_pattern = r"(def\s+\w+.*?\n\s*)(# Get state from self\._state|# Get state|# Ensure resources are initialized|# Check if already initialized)"
        content = re.sub(
            method_pattern,
            r"\1# Get state\n        state = self._state_manager.get_state()",
            content,
            flags=re.DOTALL
        )
        
        # Replace self._state with state in the rest of the method
        content = re.sub(
            r"self\._state\.(initialized|cache|model|vectorizer|error)",
            r"state.\1",
            content
        )
        
        # Write the updated content back to the file
        if content != open(file_path, "r", encoding="utf-8").read():
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated state references in {file_path}")
            return 1
        return 0
    except Exception as e:
        print(f"Error updating state references in {file_path}: {str(e)}")
        return 0


def process_file(file_path: str, component_type: str) -> None:
    """Process a file to standardize direct state initialization."""
    print(f"\nProcessing {file_path}")
    
    # Update direct state initialization
    init_updated = update_direct_state_init(file_path, component_type)
    
    # Update references to self._state
    refs_updated = update_state_references(file_path)
    
    # Summary
    if init_updated or refs_updated > 0:
        print(f"Summary for {file_path}:")
        if init_updated:
            print("  - Updated direct state initialization to use StateManager")
        if refs_updated > 0:
            print("  - Updated state references to use state = self._state_manager.get_state()")
    else:
        print(f"No changes needed for {file_path}")


def main():
    """Main function."""
    if len(sys.argv) < 3:
        print("Usage: python standardize_direct_state_init.py <base_dir> <component_type>")
        print("Component types: classifier, critic, rule, model, all")
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
        print(f"Found {len(component_files)} {comp_type} files with direct state initialization")
        
        # Process each file
        for file_path in component_files:
            process_file(file_path, comp_type)


if __name__ == "__main__":
    main()
