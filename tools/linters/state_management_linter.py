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


class StateManagementVisitor(ast.NodeVisitor):
    """AST visitor to find state management issues"""

    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[Tuple[int, str]] = []
        self.class_stack: List[str] = []
        self.current_class: Optional[str] = None
        self.current_method: Optional[str] = None
        self.found_super_call = False
        self.state_manager_instances: Set[str] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition"""
        self.class_stack.append(node.name)
        self.current_class = node.name

        # Check if class extends BaseComponent
        extends_base_component = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseComponent":
                extends_base_component = True
            elif isinstance(base, ast.Attribute) and base.attr == "BaseComponent":
                extends_base_component = True

        if extends_base_component:
            # Check for PrivateAttr with factory function
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.target.id == "_state_manager":
                        if not hasattr(item, "value") or not item.value:
                            # Missing initialization
                            self.issues.append(
                                (
                                    item.lineno,
                                    "BaseComponent subclass has _state_manager without proper initialization",
                                )
                            )
                        elif (
                            not isinstance(item.value, ast.Call)
                            or not hasattr(item.value.func, "id")
                            or not item.value.func.id.startswith("PrivateAttr")
                        ):
                            # Not using PrivateAttr
                            self.issues.append(
                                (
                                    item.lineno,
                                    "BaseComponent subclass should use PrivateAttr for _state_manager",
                                )
                            )

        # Visit all children
        self.generic_visit(node)
        self.class_stack.pop()
        if self.class_stack:
            self.current_class = self.class_stack[-1]
        else:
            self.current_class = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition"""
        self.current_method = node.name
        self.found_super_call = False

        # Check for _initialize_state method
        if self.current_class and node.name == "_initialize_state":
            # Look for super()._initialize_state() call
            for item in node.body:
                if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
                    call = item.value
                    if (
                        isinstance(call.func, ast.Attribute)
                        and call.func.attr == "_initialize_state"
                        and isinstance(call.func.value, ast.Call)
                        and hasattr(call.func.value.func, "id")
                        and call.func.value.func.id == "super"
                    ):
                        self.found_super_call = True
                        break

            # Check if super() call was found
            if not self.found_super_call:
                self.issues.append(
                    (
                        node.lineno,
                        "Missing super()._initialize_state() call in _initialize_state method",
                    )
                )

        self.generic_visit(node)
        self.current_method = None

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment statements"""
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "_state_manager":
                if isinstance(node.value, ast.Call):
                    if (
                        isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "StateManager"
                    ):
                        # Direct instantiation of StateManager
                        self.issues.append(
                            (
                                node.lineno,
                                "Direct instantiation of StateManager. Use factory functions or dependency injection instead.",
                            )
                        )
                    elif not (
                        isinstance(node.value.func, ast.Name)
                        and node.value.func.id.startswith("create_")
                        and "_state" in node.value.func.id
                    ):
                        # Not using a factory function
                        self.issues.append(
                            (
                                node.lineno,
                                "Not using a proper factory function for StateManager initialization",
                            )
                        )

                if isinstance(target.value, ast.Name):
                    self.state_manager_instances.add(target.value.id)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function/method calls"""
        # Check for state_manager.set() calls
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "set"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_state_manager"
        ):
            self.issues.append(
                (
                    node.lineno,
                    "Using non-existent set() method. Use update() for state data or set_metadata() for metadata.",
                )
            )

        # Check for self._state_manager.set() calls
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "set"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.state_manager_instances
        ):
            self.issues.append(
                (
                    node.lineno,
                    "Using non-existent set() method. Use update() for state data or set_metadata() for metadata.",
                )
            )

        self.generic_visit(node)


def check_file(filename: str) -> List[Tuple[int, str]]:
    """Check a single file for state management issues"""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        tree = ast.parse(content, filename=filename)
        visitor = StateManagementVisitor(filename)
        visitor.visit(tree)
        return visitor.issues
    except SyntaxError:
        return [(0, f"Syntax error in file {filename}")]


def find_python_files(paths: List[str]) -> List[str]:
    """Find all Python files in the given paths"""
    python_files = []

    for path in paths:
        if os.path.isfile(path) and path.endswith(".py"):
            python_files.append(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".py"):
                        python_files.append(os.path.join(root, file))

    return python_files


def main() -> int:
    """Main entry point"""
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["."]
    python_files = find_python_files(paths)

    if not python_files:
        print("No Python files found in the specified paths.")
        return 1

    total_issues = 0
    for filename in python_files:
        issues = check_file(filename)
        if issues:
            print(f"\n{filename}:")
            for line, message in issues:
                print(f"  Line {line}: {message}")
                total_issues += 1

    if total_issues:
        print(f"\nFound {total_issues} state management issues.")
        return 1
    else:
        print("No state management issues found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
