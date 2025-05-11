#!/usr/bin/env python
"""
Standardization Script for Sifaka

This script analyzes the Sifaka codebase to identify inconsistencies in implementation patterns
and suggests fixes to standardize them across all components.

Usage:
    python scripts/standardize_patterns.py [--fix] [--component COMPONENT]

Options:
    --fix               Apply suggested fixes automatically
    --component COMPONENT   Only analyze/fix the specified component
"""

import os
import re
import sys
import argparse
import importlib
import inspect
from typing import Dict, List, Set, Tuple, Optional, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Sifaka utilities
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Define patterns to check
PATTERNS = {
    "state_management": {
        "correct": r"self\._state_manager\.(update|get|set_metadata)",
        "incorrect": r"self\._state\[",
        "fix": lambda code: code.replace("self._state[", "self._state_manager.update(")
    },
    "error_handling": {
        "correct": r"safely_execute_(component|chain|model|rule|critic|classifier|retrieval)_operation",
        "incorrect": r"try:(?:(?!safely_execute).)*?except\s+Exception",
        "fix": lambda code: re.sub(
            r"try:(?P<body>(?:(?!safely_execute).)*?)except\s+Exception\s+as\s+(?P<var>\w+):",
            r"def operation():\n\g<body>\n\nresult = safely_execute_component_operation(\n    operation=operation,\n    component_name=self.name,\n    component_type=self.__class__.__name__\n)",
            code,
            flags=re.DOTALL
        )
    },
    "pattern_matching": {
        "correct": r"from\s+sifaka\.utils\.patterns\s+import",
        "incorrect": r"import\s+re.*?re\.(compile|match|search|sub|findall)",
        "fix": lambda code: re.sub(
            r"import\s+re",
            r"from sifaka.utils.patterns import compile_pattern, match_pattern, find_patterns, replace_pattern",
            code
        )
    },
    "factory_function": {
        "correct": r"def\s+create_\w+\([^)]*\):\s*\"\"\"[^\"]*\"\"\"",
        "incorrect": r"def\s+create_\w+\([^)]*\):\s*(?!\"\"\")",
        "fix": lambda code, name: re.sub(
            r"def\s+(create_\w+)(\([^)]*\)):\s*(?!\"\"\")",
            r"def \1\2:\n    \"\"\"\n    Create a component with the given configuration.\n    \n    Args:\n        # TODO: Document parameters\n        \n    Returns:\n        # TODO: Document return type\n        \n    Raises:\n        ValueError: If configuration is invalid\n        TypeError: If input types are incompatible\n    \"\"\"\n",
            code
        )
    },
    "lifecycle_management": {
        "correct": r"def\s+(warm_up|cleanup|initialize)\s*\(\s*self\s*\)",
        "incorrect": r"class\s+\w+.*?(?!def\s+(?:warm_up|cleanup|initialize)\s*\(\s*self\s*\))",
        "fix": lambda code: code  # This requires manual fixing
    }
}

def analyze_file(file_path: str) -> Dict[str, List[Tuple[int, str]]]:
    """
    Analyze a file for pattern inconsistencies.
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        Dictionary of pattern names to lists of (line number, line content) tuples
    """
    with open(file_path, "r") as f:
        content = f.read()
        lines = content.split("\n")
    
    issues = {}
    
    for pattern_name, pattern_info in PATTERNS.items():
        # Check for correct pattern
        correct_matches = re.finditer(pattern_info["correct"], content, re.DOTALL)
        correct_lines = set(lines.index(lines[i]) for match in correct_matches 
                          for i in range(len(lines)) if match.start() <= len("\n".join(lines[:i+1])) <= match.end())
        
        # Check for incorrect pattern
        incorrect_matches = re.finditer(pattern_info["incorrect"], content, re.DOTALL)
        incorrect_lines = [(lines.index(lines[i]), lines[i]) for match in incorrect_matches 
                          for i in range(len(lines)) if match.start() <= len("\n".join(lines[:i+1])) <= match.end()]
        
        # Filter out lines that also match the correct pattern
        incorrect_lines = [(line_num, line) for line_num, line in incorrect_lines 
                          if line_num not in correct_lines]
        
        if incorrect_lines:
            issues[pattern_name] = incorrect_lines
    
    return issues

def fix_file(file_path: str, issues: Dict[str, List[Tuple[int, str]]]) -> bool:
    """
    Fix pattern inconsistencies in a file.
    
    Args:
        file_path: Path to the file to fix
        issues: Dictionary of pattern names to lists of (line number, line content) tuples
        
    Returns:
        True if changes were made, False otherwise
    """
    with open(file_path, "r") as f:
        content = f.read()
    
    original_content = content
    
    for pattern_name, pattern_issues in issues.items():
        pattern_info = PATTERNS[pattern_name]
        fix_func = pattern_info["fix"]
        
        # Apply the fix function to the content
        content = fix_func(content)
    
    # Write the fixed content back to the file if changes were made
    if content != original_content:
        with open(file_path, "w") as f:
            f.write(content)
        return True
    
    return False

def find_python_files(directory: str) -> List[str]:
    """
    Find all Python files in a directory and its subdirectories.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of paths to Python files
    """
    python_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    return python_files

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Standardize implementation patterns in Sifaka")
    parser.add_argument("--fix", action="store_true", help="Apply suggested fixes automatically")
    parser.add_argument("--component", help="Only analyze/fix the specified component")
    
    args = parser.parse_args()
    
    # Determine which directories to analyze
    if args.component:
        directories = [f"sifaka/{args.component}"]
    else:
        directories = ["sifaka"]
    
    # Find all Python files
    python_files = []
    for directory in directories:
        if os.path.exists(directory):
            python_files.extend(find_python_files(directory))
        else:
            logger.error(f"Directory not found: {directory}")
    
    # Analyze files
    total_issues = 0
    files_with_issues = 0
    
    for file_path in python_files:
        issues = analyze_file(file_path)
        
        if issues:
            files_with_issues += 1
            issue_count = sum(len(pattern_issues) for pattern_issues in issues.values())
            total_issues += issue_count
            
            logger.info(f"Found {issue_count} issues in {file_path}")
            
            for pattern_name, pattern_issues in issues.items():
                logger.info(f"  {pattern_name}: {len(pattern_issues)} issues")
                
                for line_num, line in pattern_issues:
                    logger.info(f"    Line {line_num + 1}: {line.strip()}")
            
            if args.fix:
                if fix_file(file_path, issues):
                    logger.info(f"Fixed issues in {file_path}")
                else:
                    logger.info(f"No automatic fixes applied to {file_path}")
    
    # Print summary
    logger.info(f"Found {total_issues} issues in {files_with_issues} files")
    
    if args.fix:
        logger.info("Applied automatic fixes where possible")
    else:
        logger.info("Run with --fix to apply automatic fixes")

if __name__ == "__main__":
    main()
