#!/usr/bin/env python3
"""
State Management Linter for Sifaka

This script checks for common state management anti-patterns in the Sifaka codebase.
It helps ensure consistent state management practices across the project.

Usage:
    python tools/linters/state_management_linter.py [paths...]

Example:
    python tools/linters/state_management_linter.py sifaka/
    python tools/linters/state_management_linter.py sifaka/critics/

Rules:
- No direct StateManager instantiation
- No use of set() method (use update() instead)
- Components should check initialization
- BaseComponent subclasses should call super()._initialize_state()
"""

import ast
import os
import re
import sys
from typing import List, Tuple, Dict, Any, Optional, Set

# Define root classes that don't need to call super()._initialize_state()
# These are the base implementations that have no parent to call
ROOT_CLASSES = {
    ("sifaka/core/base.py", "BaseComponent"),
    ("sifaka/core/initialization.py", "InitializableMixin"),
}

# Files that are allowed to use direct StateManager instantiation
# since they define or implement core state functionality
ALLOWED_DIRECT_INSTANTIATION_FILES = ["sifaka/utils/state.py"]

# Files that are allowed to import StateManager without a factory function
# since they are root implementations or need direct access to StateManager
# Note: These files should eventually be fixed, but we're focusing on critical paths first
ALLOWED_IMPORT_FILES = [
    "sifaka/utils/state.py",  # Defines StateManager
    "sifaka/core/base.py",  # Core component implementation
    "sifaka/core/initialization.py",  # Core initialization
    "sifaka/utils/common.py",
]


class StateManagementVisitor(ast.NodeVisitor):
    """
    AST visitor that checks for state management anti-patterns.
    """

    def __init__(self, filename: str) -> None:
        """Initialize the visitor."""
        self.violations: List[Tuple[int, str]] = []
        self.filename = filename
        self.current_class: Optional[str] = None
        self.has_initialize_state = False
        self.calls_super_initialize_state = False

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        """Visit a class definition."""
        old_class = self.current_class
        self.current_class = node.name
        self.has_initialize_state = False
        self.calls_super_initialize_state = False

        # Visit all the class methods
        self.generic_visit(node)

        # Reset the class context when done
        self.current_class = old_class
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Visit a function definition."""
        if node.name == "_initialize_state":
            self.has_initialize_state = True

            # Check if super()._initialize_state() is called
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if (
                        isinstance(child.func.value, ast.Call)
                        and isinstance(child.func.value.func, ast.Name)
                        and child.func.value.func.id == "super"
                        and child.func.attr == "_initialize_state"
                    ):
                        self.calls_super_initialize_state = True
                        break

            # Check if this is a root class that doesn't need to call super()
            is_root_class = False
            for path, class_name in ROOT_CLASSES:
                if self.filename.endswith(path) and self.current_class == class_name:
                    is_root_class = True
                    break

            # Report violation if not a root class and doesn't call super()
            if not self.calls_super_initialize_state and not is_root_class:
                self.violations.append(
                    (
                        node.lineno,
                        f"Missing super()._initialize_state() call in _initialize_state method",
                    )
                )

        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        """Visit an assignment."""
        # Check for direct StateManager instantiation, unless in an allowed file
        for allowed_file in ALLOWED_DIRECT_INSTANTIATION_FILES:
            if self.filename.endswith(allowed_file):
                return self.generic_visit(node)

        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "StateManager":
                self.violations.append(
                    (
                        node.lineno,
                        f"Direct StateManager instantiation detected. Use create_*_state() factory functions instead.",
                    )
                )
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        """Visit a function call."""
        # Check for set() method calls on state manager
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "set"
            and not node.func.attr == "set_metadata"
        ):
            # Check if it's a state manager call
            if (
                isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "_state_manager"
            ):
                self.violations.append(
                    (node.lineno, f"Using set() method on state manager. Use update() instead.")
                )
        return self.generic_visit(node)


def lint_file(filepath: str) -> List[Tuple[int, str]]:
    """
    Lint a single file for state management issues.

    Args:
        filepath: The path to the file to lint

    Returns:
        A list of (line number, issue message) tuples
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        visitor = StateManagementVisitor(filepath)
        visitor.visit(tree)

        # Also check for StateManager imports
        if "from sifaka.utils.state import StateManager" in source:
            # Skip import check for allowed files
            if not any(filepath.endswith(f) for f in ALLOWED_IMPORT_FILES):
                # Make sure it also imports a factory function
                pattern = r"from sifaka\.utils\.state import .*create_\w+_state"
                if not re.search(pattern, source):
                    visitor.violations.append(
                        (1, "Importing StateManager without importing a factory function")
                    )

        return visitor.violations
    except Exception as e:
        return [(0, f"Error linting {filepath}: {str(e)}")]


def lint_directory(dirpath: str) -> Dict[str, List[Tuple[int, str]]]:
    """
    Recursively lint files in a directory.

    Args:
        dirpath: The directory to lint

    Returns:
        A dict mapping filenames to lists of (line number, issue message) tuples
    """
    results: Dict[str, List[Tuple[int, str]]] = {}

    for root, _, files in os.walk(dirpath):
        for filename in files:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(root, filename)
            issues = lint_file(filepath)

            if issues:
                results[filepath] = issues

    return results


def main() -> int:
    """
    Main function.

    Returns:
        Exit code
    """
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} [paths...]")
        return 1

    paths = sys.argv[1:]
    all_results: Dict[str, List[Tuple[int, str]]] = {}

    for path in paths:
        if os.path.isdir(path):
            results = lint_directory(path)
        elif os.path.isfile(path) and path.endswith(".py"):
            results = {path: lint_file(path)}
        else:
            print(f"Skipping {path} (not a Python file or directory)")
            continue

        all_results.update(results)

    # Print results
    issue_count = 0
    for filepath, issues in sorted(all_results.items()):
        if issues:
            print(f"\n{filepath}:")
            for line, message in sorted(issues):
                print(f"  Line {line}: {message}")
                issue_count += 1

    if issue_count:
        print(f"\nFound {issue_count} state management issues.")
        return 1
    else:
        print("No state management issues found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
